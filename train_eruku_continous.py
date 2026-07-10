import torch
import wandb
import argparse
from pathlib import Path
from tqdm import tqdm
from einops import rearrange
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import webdataset as wds

from accelerate import Accelerator, DistributedDataParallelKwargs
from custom_datasets import dataset_factory
from eruku_continuous_inf import DDPCompatibleEmuru
from hwd.datasets.shtg import KaraokeLines
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


def karaoke_collate_fn(batch):
    out = {}
    for key in batch[0]:
        values = [d[key] for d in batch]
        # Stack images/tensors
        if key in ['style_img', 'gen_img']:
            tensorized = []
            for v in values:
                if isinstance(v, torch.Tensor):
                    tensorized.append(v)
                else:  # Assume PIL Image
                    tensorized.append(transforms.ToTensor()(v))
            out[key] = torch.stack(tensorized)
        elif isinstance(values[0], (str, int, float)):
            out[key] = values
        elif isinstance(values[0], (list, tuple)):
            if values[0] and isinstance(values[0][0], Path):
                out[key] = [[str(p) for p in v] for v in values]
            else:
                out[key] = values
        elif isinstance(values[0], Path):
            out[key] = [str(v) for v in values]
        else:
            out[key] = values
    # Alias for compatibility with IAM evaluation code
    out['same_img'] = out['gen_img']
    out['same_text'] = out['gen_text']
    out['style_text'] = out['style_imgs_text']
    return out

class Config:
    """Training configuration"""
    # Model and training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.01
    MAX_EPOCHS = 2500
    MAX_IMG_LEN = 8192
    
    # Data paths
    DATASETS_ROOT = '/mnt/nas/datasets/'
    TRAIN_PATTERN = "/scratch/project_465002151/font-square-pretrain-20M/{000000..002291}.tar"
    CHECKPOINT_DIR = "/scratch/project_465002151/eruku_checkpoints"
    
    # Training settings
    NUM_WORKERS = 1
    RANDOM_SEED = 25042025
    STYLE_TEXT_DROPOUT_PROB = 0.1  # Probability of replacing style text with empty string during training
    
    # Wandb and checkpointing
    PROJECT_NAME = 'Eruku_continuous'
    CHECKPOINT_SAVE_FREQUENCY = 1  # Save every N epochs
    LOG_CHECKPOINT_SAMPLES = 48000 # Log and checkpoint every N samples

    # CFG settings
    GEN_TEXT_DROPOUT_PROB = 0.05
    CFG_SCALES = [1, 1.5, 1.75, 2.0, 2.25]
    PERFORM_CFG = False


class DataProcessor:
    """Handles data loading and processing operations"""
    
    @staticmethod
    def pad_images(images, padding_value=1):
        """Pad images to same width for batching"""
        images = [rearrange(img, 'c h w -> w c h') for img in images]
        return rearrange(pad_sequence(images, padding_value=padding_value), 'w b c h -> b c h w')
    
    @staticmethod
    def convert_to_tensor(img):
        """Convert PIL Image to tensor and ensure 3 channels"""
        if hasattr(img, 'shape'):  # Already a tensor
            tensor = img
        else:  # PIL Image
            tensor = transforms.ToTensor()(img)
        
        # Convert grayscale to RGB if needed
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        
        return tensor
    
    @staticmethod
    def pad_images_fixed(images, max_width=Config.MAX_IMG_LEN, padding_value=1):
        """Pad images to fixed maximum width"""
        padded_images = []
        for img in images:
            c, h, w = img.shape
            if w > max_width:
                img = img[:, :, :max_width]  # Crop if too wide
                w = max_width
            
            if w < max_width:
                pad_width = max_width - w
                img = torch.nn.functional.pad(img, (0, pad_width), value=padding_value)
            
            padded_images.append(img)
        
        return torch.stack(padded_images)
    
    @staticmethod
    def text_to_tensor(text_list, max_text_len=256):
        """Convert text to padded byte tensors for DDP compatibility"""
        byte_tensors = []
        for text in text_list:
            byte_data = text.encode('utf-8')
            byte_tensor = torch.tensor(list(byte_data), dtype=torch.uint8)
            byte_tensors.append(byte_tensor)
        
        # Pad to fixed length
        padded_bytes = []
        for tensor in byte_tensors:
            if len(tensor) > max_text_len:
                tensor = tensor[:max_text_len]
            elif len(tensor) < max_text_len:
                pad_size = max_text_len - len(tensor)
                tensor = torch.nn.functional.pad(tensor, (0, pad_size), value=0)
            padded_bytes.append(tensor)
        
        return torch.stack(padded_bytes)
    
    @staticmethod
    def tensor_to_text(byte_tensor):
        """Convert byte tensors back to text strings"""
        texts = []
        for row in byte_tensor:
            non_zero_bytes = row[row != 0].cpu().numpy()
            try:
                text = bytes(non_zero_bytes).decode('utf-8')
                if not text.strip():
                    text = "sample text"  # Default for empty strings
                texts.append(text)
            except UnicodeDecodeError:
                texts.append("sample text")  # Default for decode errors
        return texts
    
    @staticmethod
    def validate_widths(style_width, gen_width):
        """Validate and fix width values to prevent DDP issues"""
        style_width = min(style_width, Config.MAX_IMG_LEN // 2)
        remaining_width = Config.MAX_IMG_LEN - style_width
        gen_width = min(gen_width, remaining_width)
        
        # Ensure minimum widths
        style_width = max(style_width, 64)
        gen_width = max(gen_width, 64)
        
        # Ensure total doesn't exceed max
        if style_width + gen_width > Config.MAX_IMG_LEN:
            total = style_width + gen_width
            style_width = int(style_width * Config.MAX_IMG_LEN / total)
            gen_width = Config.MAX_IMG_LEN - style_width
        
        return style_width, gen_width


def collate_pairs_hf(batch):
    """Collate function for HuggingFace webdataset format"""
    processor = DataProcessor()
    
    # Extract images and convert to tensors
    style_imgs = [processor.convert_to_tensor(sample['style.bw.png']) for sample in batch]
    gen_imgs = [processor.convert_to_tensor(sample['gen.bw.png']) for sample in batch]
    
    # Pad images to fixed size
    style_imgs_padded = processor.pad_images_fixed(style_imgs)
    gen_imgs_padded = processor.pad_images_fixed(gen_imgs)
    
    # Extract and process text
    style_texts = [sample['json']['style_text'] for sample in batch]
    gen_texts = [sample['json']['gen_text'] for sample in batch]
    
    # Calculate and validate width values
    style_widths = []
    gen_widths = []
    
    for sample in batch:
        style_width = min(sample['json']['style_img_width'], Config.MAX_IMG_LEN // 2)
        remaining_width = Config.MAX_IMG_LEN - style_width
        gen_width = min(sample['json']['gen_img_width'], remaining_width)
        
        style_width, gen_width = processor.validate_widths(style_width, gen_width)
        style_widths.append(style_width)
        gen_widths.append(gen_width)
    
    return {
        'style_img': style_imgs_padded,
        'gen_img': gen_imgs_padded,
        'style_img_width': torch.tensor(style_widths),
        'gen_img_width': torch.tensor(gen_widths),
        'style_text_bytes': processor.text_to_tensor(style_texts),
        'gen_text_bytes': processor.text_to_tensor(gen_texts),
    }


class DataLoaderManager:
    """Handles dataset creation and data loading"""
    
    def __init__(self, config: Config):
        self.config = config
        self.processor = DataProcessor()
        
    def create_train_dataset(self):
        """Create training dataset using WebDataset"""
        transform = transforms.ToTensor()
        
        dataset = (
            wds.WebDataset(self.config.TRAIN_PATTERN,  nodesplitter=wds.split_by_node,
                          shardshuffle=100)
            .decode("pil")
            .map(lambda sample: {
                "style.rgb.png": transform(sample["style.rgb.png"]),
                "style.bw.png": transform(sample["style.bw.png"]),
                "gen.rgb.png": transform(sample["gen.rgb.png"]),
                "gen.bw.png": transform(sample["gen.bw.png"]),
                'json': sample['json']
            })
        )
        
        return DataLoader(
            dataset, 
            batch_size=self.config.BATCH_SIZE,
            pin_memory=True,
            collate_fn=collate_pairs_hf,
            num_workers=self.config.NUM_WORKERS,
            drop_last=True,
            persistent_workers=True
        )
    
    def create_eval_dataset(self):
        """Create evaluation dataset with Karaoke datasets"""
        karaoke_handw = KaraokeLines('handwritten', num_style_samples=1, load_gen_sample=True)
        karaoke_typew = KaraokeLines('typewritten', num_style_samples=1, load_gen_sample=True)
        eval_dataset = ConcatDataset([SHTGWrapper(karaoke_handw), SHTGWrapper(karaoke_typew)])

        return DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=karaoke_collate_fn,
            num_workers=1,
            persistent_workers=False
        )


class TrainingManager:
    """Manages the training process"""
    
    def __init__(self, config: Config):
        self.config = config
        self.processor = DataProcessor()
        self.global_step = 0
        self.samples_processed = 0
        self.running_loss = 0.0
        self.running_mse_loss = 0.0
        self.running_ce_loss = 0.0
        self.running_ocr_loss = 0.0
        self.steps_since_log = 0
        self.style_dropout_rate = 0.0
        
        # Calculate gradient accumulation to always accumulate to 256 samples total
        # Get number of processes from environment or default to 1
        import os
        num_processes = int(os.environ.get('WORLD_SIZE', '1'))
        
        # Calculate accumulation steps: 256 total samples / (batch_size_per_gpu * num_gpus)
        samples_per_iteration = config.BATCH_SIZE * num_processes
        target_batch_size = 256
        gradient_accumulation_steps = max(1, target_batch_size // samples_per_iteration)
        
        if num_processes == 1:
            print(f"Single GPU training: batch_size={config.BATCH_SIZE}, "
                  f"accumulation_steps={gradient_accumulation_steps}, "
                  f"effective_batch_size={config.BATCH_SIZE * gradient_accumulation_steps}")
        else:
            print(f"Multi-GPU training: {num_processes} GPUs, batch_size_per_gpu={config.BATCH_SIZE}, "
                  f"accumulation_steps={gradient_accumulation_steps}, "
                  f"effective_batch_size={samples_per_iteration * gradient_accumulation_steps}")
        
        # Initialize accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs]
        )
        
    def setup_model_and_optimizer(self):
        """Initialize model and optimizer"""
        # Set random seed
        torch.manual_seed(self.config.RANDOM_SEED)
        random.seed(self.config.RANDOM_SEED)
        
        # Create model
        self.model = DDPCompatibleEmuru("google-t5/t5-large", 'blowing-up-groundhogs/emuru_vae').cuda()
        self.model.print_parameter_info()
        
        # Create optimizer
        trainable_params = self.model.get_trainable_parameters()
        print(f"Optimizing {len(trainable_params)} parameter groups")
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
    def setup_data_loaders(self):
        """Setup training and evaluation data loaders"""
        data_loader = DataLoaderManager(self.config)
        self.train_loader = data_loader.create_train_dataset()
        self.eval_loader = data_loader.create_eval_dataset()
        
    def prepare_for_training(self):
        """Prepare model, optimizer, and data loaders with accelerator"""
        self.model, self.optimizer, self.train_loader, self.eval_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.eval_loader
        )
        
    def setup_wandb(self, run_name, wandb_id=None):
        """Setup wandb for logging"""
        if self.accelerator.is_main_process:
            wandb.init(
                project=self.config.PROJECT_NAME,
                name=run_name,
                # id=wandb_id,
                resume='allow',
                config=vars(self.config),
            )
    
    def load_checkpoint(self, checkpoint_dir, override_checkpoint_run_name=None):
        """Load checkpoint if available"""
        start_epoch = 0
        wandb_id = None
        
        actual_checkpoint_dir = checkpoint_dir
        if override_checkpoint_run_name:
            actual_checkpoint_dir = f"{self.config.CHECKPOINT_DIR}/{override_checkpoint_run_name}/"
        
        try:
            checkpoint_path = sorted(Path(actual_checkpoint_dir).rglob('*.pth'))[-1]
            checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
            
            # Load model state dict
            model_state_dict = checkpoint['model']
            
            # Handle DDP prefix differences
            if not hasattr(self.model, 'module') and any(k.startswith('module.') for k in model_state_dict.keys()):
                model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
            elif hasattr(self.model, 'module') and not any(k.startswith('module.') for k in model_state_dict.keys()):
                model_state_dict = {f'module.{k}': v for k, v in model_state_dict.items()}
            
            self.model.load_state_dict(model_state_dict, strict=False)
            
            # Load optimizer state
            if 'optimizer' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    print("Loaded optimizer state")
                except Exception as e:
                    print(f"Warning: Could not load optimizer state: {e}")
            
            start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint.get('global_step', 0)
            self.samples_processed = checkpoint.get('samples_processed', 0)
            
            # Fix for old checkpoints that don't have samples_processed
            if self.samples_processed == 0 and self.global_step > 0:
                # Each global step processes effective_batch_size samples
                samples_per_iteration = self.config.BATCH_SIZE * self.accelerator.num_processes
                effective_batch_size = samples_per_iteration * self.accelerator.gradient_accumulation_steps
                estimated_samples = self.global_step * effective_batch_size
                self.samples_processed = estimated_samples
                if self.accelerator.is_main_process:
                    print(f"Old checkpoint detected. Estimated samples_processed: {self.samples_processed} "
                          f"(global_step={self.global_step} × effective_batch_size={effective_batch_size})")
            
            if not override_checkpoint_run_name:
                wandb_id = checkpoint['wandb_id']
            
            print(f'Resumed training from {checkpoint_path} at epoch {start_epoch} (global_step={self.global_step})')
            del checkpoint
            
        except Exception as e:
            print(f"No checkpoint found or error loading: {e}")
            print("Starting training from scratch")
        
        return start_epoch, wandb_id
    
    def process_batch(self, sample):
        """Process a single batch and convert tensors"""
        # Move tensors to device using accelerator
        sample = {k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
        
        # Convert byte tensors back to text
        sample['style_text'] = self.processor.tensor_to_text(sample['style_text_bytes'])
        sample['gen_text'] = self.processor.tensor_to_text(sample['gen_text_bytes'])
        
        # Apply style text dropout during training
        if self.model.training and self.config.STYLE_TEXT_DROPOUT_PROB > 0:
            dropout_mask = torch.rand(len(sample['style_text']), device=self.accelerator.device) < self.config.STYLE_TEXT_DROPOUT_PROB
            sample['style_text_dropped'] = dropout_mask.tolist()  # Track which samples have dropped style text
            
            # Replace style text with empty string for selected samples
            for i, should_drop in enumerate(dropout_mask):
                if should_drop:
                    sample['style_text'][i] = ""
        else:
            # During evaluation, no dropout
            sample['style_text_dropped'] = [False] * len(sample['style_text'])

        # Clean up byte tensors
        del sample['style_text_bytes']
        del sample['gen_text_bytes']
        
        return sample
    
    def get_model_inputs(self, sample):
        """Get model inputs from processed sample"""
        if hasattr(self.model, 'module'):
            return self.model.module.module_get_model_inputs(
                sample['style_img'], sample['gen_img'],
                sample['style_img_width'], sample['gen_img_width'],
                self.config.MAX_IMG_LEN
            )
        else:
            return self.model.get_model_inputs(
                sample['style_img'], sample['gen_img'],
                sample['style_img_width'], sample['gen_img_width'],
                self.config.MAX_IMG_LEN
            )
    
    def training_step(self, sample, model_inputs):
        """Perform a single training step"""
        model_out = self.model({**sample, **model_inputs}, mode='train')
        loss = model_out[0]['loss']
        mse_loss = model_out[0]['mse_loss']
        ce_loss = model_out[0]['ce_loss']
        ocr_loss = model_out[0]['ocr_loss']

        self.accelerator.backward(loss)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return model_out, loss, mse_loss, ce_loss, ocr_loss
    
    def evaluation_step(self):
        """Perform distributed evaluation on the validation set"""
        self.model.eval()
        eval_loss_sum = 0
        eval_mse_loss_sum = 0
        eval_ce_loss_sum = 0
        eval_ocr_loss_sum = 0

        # Separate metrics for samples without style text
        eval_loss_no_style_sum = 0
        eval_mse_loss_no_style_sum = 0
        eval_ce_loss_no_style_sum = 0
        eval_ocr_loss_no_style_sum = 0

        total_batches = 0
        total_no_style_batches = 0
        
        last_batch = None
        last_embeddings = None
        last_model_out = None
        last_batch_no_style = None
        last_embeddings_no_style = None
        last_model_out_no_style = None
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc='Evaluation', disable=not self.accelerator.is_main_process):
                # Validate widths for evaluation batch
                raw_style_width = batch['style_img'].shape[-1]
                raw_gen_width = batch['same_img'].shape[-1]
                validated_style_width, validated_gen_width = self.processor.validate_widths(
                    raw_style_width, raw_gen_width
                )
                
                # Prepare evaluation batch with style text
                eval_batch = {
                    'style_img': batch['style_img'],
                    'gen_img': batch['same_img'],
                    'style_img_width': validated_style_width,
                    'gen_img_width': validated_gen_width,
                    'style_text': batch['style_text'],
                    'gen_text': batch['same_text'],
                    'style_text_dropped': [False] * len(batch['style_text'])
                }
                
                # Get model inputs and run forward pass
                embeddings = self.get_model_inputs(eval_batch)
                model_out = self.model({**eval_batch, **embeddings}, mode='train')
                
                # Gather losses from all processes
                loss = self.accelerator.gather(model_out[0]['loss']).mean()
                mse_loss = self.accelerator.gather(model_out[0]['mse_loss']).mean()
                ce_loss = self.accelerator.gather(model_out[0]['ce_loss']).mean()
                ocr_loss = self.accelerator.gather(model_out[0]['ocr_loss']).mean()

                eval_loss_sum += loss
                eval_mse_loss_sum += mse_loss
                eval_ce_loss_sum += ce_loss
                eval_ocr_loss_sum += ocr_loss
                total_batches += 1
                
                # Keep last batch for visualization (only on main process)
                if self.accelerator.is_main_process:
                    last_batch = eval_batch
                    last_embeddings = embeddings
                    last_model_out = model_out
                
                # Evaluate same batch without style text
                eval_batch_no_style = {
                    'style_img': batch['style_img'],
                    'gen_img': batch['same_img'],
                    'style_img_width': validated_style_width,
                    'gen_img_width': validated_gen_width,
                    'style_text': [""] * len(batch['style_text']),  # Empty style text
                    'gen_text': batch['same_text'],
                    'style_text_dropped': [True] * len(batch['style_text'])
                }
                
                # Get model inputs and run forward pass for no-style version
                embeddings_no_style = self.get_model_inputs(eval_batch_no_style)
                model_out_no_style = self.model({**eval_batch_no_style, **embeddings_no_style}, mode='train')
                
                # Gather losses from all processes for no-style version
                loss_no_style = self.accelerator.gather(model_out_no_style[0]['loss']).mean()
                mse_loss_no_style = self.accelerator.gather(model_out_no_style[0]['mse_loss']).mean()
                ce_loss_no_style = self.accelerator.gather(model_out_no_style[0]['ce_loss']).mean()
                ocr_loss_no_style = self.accelerator.gather(model_out_no_style[0]['ocr_loss']).mean()

                eval_loss_no_style_sum += loss_no_style
                eval_mse_loss_no_style_sum += mse_loss_no_style
                eval_ce_loss_no_style_sum += ce_loss_no_style
                eval_ocr_loss_no_style_sum += ocr_loss_no_style
                total_no_style_batches += 1
                
                # Keep last no-style batch for visualization (only on main process)
                if self.accelerator.is_main_process:
                    last_batch_no_style = eval_batch_no_style
                    last_embeddings_no_style = embeddings_no_style
                    last_model_out_no_style = model_out_no_style
        
        self.model.train()
        
        # Average losses across all batches
        return {
            'eval_loss': eval_loss_sum / total_batches,
            'eval_mse_loss': eval_mse_loss_sum / total_batches,
            'eval_ce_loss': eval_ce_loss_sum / total_batches,
            'eval_ocr_loss': eval_ocr_loss_sum / total_batches,
            'eval_loss_no_style': eval_loss_no_style_sum / total_no_style_batches,
            'eval_mse_loss_no_style': eval_mse_loss_no_style_sum / total_no_style_batches,
            'eval_ce_loss_no_style': eval_ce_loss_no_style_sum / total_no_style_batches,
            'eval_ocr_loss': eval_ocr_loss_sum / total_batches,
            'eval_ocr_loss_no_style': eval_ocr_loss_no_style_sum / total_no_style_batches,
            'last_batch': last_batch,
            'last_embeddings': last_embeddings,
            'last_model_out': last_model_out,
            'last_batch_no_style': last_batch_no_style,
            'last_embeddings_no_style': last_embeddings_no_style,
            'last_model_out_no_style': last_model_out_no_style
        }
    
    def create_visualization_images(self, sample, model_inputs, model_out, eval_results):
        """Create images for wandb logging"""
        wandb_data = {}
        rank_info = f" [GPU {self.accelerator.process_index}]" if self.accelerator.num_processes > 1 else ""
        
        # Training visualization
        model_latent = model_out[1]
        
        if hasattr(self.model, 'module'):
            output = torch.clamp(self.model.module.module_vae_decode(model_latent).sample, 0, 1)
            gt_latent = rearrange(model_inputs['decoder_inputs_embeds'], 'b w (h c) -> b c h w',
                                b=model_latent.shape[0], h=8, c=1)
            input_img = torch.clamp(self.model.module.module_vae_decode(gt_latent).sample, 0, 1)
        else:
            output = torch.clamp(self.model.vae.decode(model_latent).sample, 0, 1)
            gt_latent = rearrange(model_inputs['decoder_inputs_embeds'], 'b w (h c) -> b c h w',
                                b=model_latent.shape[0], h=8, c=1)
            input_img = torch.clamp(self.model.vae.decode(gt_latent).sample, 0, 1)
        
        out = torch.cat([input_img.repeat(1, 3, 1, 1), output.repeat(1, 3, 1, 1)], dim=-1)
        
        # Check if any samples in the batch have style text dropped
        has_style_dropped = any(sample.get('style_text_dropped', [False]))
        
        # Create separate visualizations for samples with and without style text
        if has_style_dropped:
            # Find samples with and without style text
            with_style_indices = [i for i, dropped in enumerate(sample.get('style_text_dropped', [])) if not dropped]
            without_style_indices = [i for i, dropped in enumerate(sample.get('style_text_dropped', [])) if dropped]
            
            if with_style_indices:
                # Build sub-batch for with style
                sub_sample = {k: (v[with_style_indices] if isinstance(v, torch.Tensor) and v.shape[0] == len(sample['style_text']) else [v[i] for i in with_style_indices] if isinstance(v, list) and len(v) == len(sample['style_text']) else v) for k, v in sample.items()}
                sub_model_inputs = self.get_model_inputs(sub_sample)
                if hasattr(self.model, 'module'):
                    synth_gen_test_with_style = self.model.module.module_continue_gen_test(
                        input_img[with_style_indices], {**sub_sample, **sub_model_inputs})
                else:
                    synth_gen_test_with_style = self.model.continue_gen_test(
                        input_img[with_style_indices], {**sub_sample, **sub_model_inputs})
                idx = 0
                wandb_data['synth_img_with_style'] = wandb.Image(
                    make_grid(out[with_style_indices][idx:idx+1], nrow=1, normalize=True),
                    caption=f"WITH STYLE - Style: {sub_sample['style_text'][idx]}, Gen: {sub_sample['gen_text'][idx]}{rank_info}"
                )
                wandb_data['synth_gen_test_with_style'] = wandb.Image(
                    synth_gen_test_with_style[idx:idx+1] if synth_gen_test_with_style.dim() == 4 else synth_gen_test_with_style,
                    caption=f"WITH STYLE - Style: {sub_sample['style_text'][idx]}, Gen: {sub_sample['gen_text'][idx]}{rank_info}"
                )
            
            if without_style_indices:
                # Build sub-batch for without style
                sub_sample = {k: (v[without_style_indices] if isinstance(v, torch.Tensor) and v.shape[0] == len(sample['style_text']) else [v[i] for i in without_style_indices] if isinstance(v, list) and len(v) == len(sample['style_text']) else v) for k, v in sample.items()}
                sub_model_inputs = self.get_model_inputs(sub_sample)
                if hasattr(self.model, 'module'):
                    synth_gen_test_without_style = self.model.module.module_continue_gen_test(
                        input_img[without_style_indices], {**sub_sample, **sub_model_inputs})
                else:
                    synth_gen_test_without_style = self.model.continue_gen_test(
                        input_img[without_style_indices], {**sub_sample, **sub_model_inputs})
                idx = 0
                wandb_data['synth_img_without_style'] = wandb.Image(
                    make_grid(out[without_style_indices][idx:idx+1], nrow=1, normalize=True),
                    caption=f"WITHOUT STYLE - Style: {sub_sample['style_text'][idx]}, Gen: {sub_sample['gen_text'][idx]}{rank_info}"
                )
                wandb_data['synth_gen_test_without_style'] = wandb.Image(
                    synth_gen_test_without_style[idx:idx+1] if synth_gen_test_without_style.dim() == 4 else synth_gen_test_without_style,
                    caption=f"WITHOUT STYLE - Style: {sub_sample['style_text'][idx]}, Gen: {sub_sample['gen_text'][idx]}{rank_info}"
                )
        else:
            # Regular visualization when no style text is dropped
            if hasattr(self.model, 'module'):
                synth_gen_test = self.model.module.module_continue_gen_test(input_img, {**sample, **model_inputs})
            else:
                synth_gen_test = self.model.continue_gen_test(input_img, {**sample, **model_inputs})
            wandb_data['synth_img'] = wandb.Image(
                make_grid(out, nrow=1, normalize=True),
                caption=f"Style: {'|'.join(sample['style_text'])}, Gen: {'|'.join(sample['gen_text'])}{rank_info}"
            )
            wandb_data['synth_gen_test'] = wandb.Image(
                synth_gen_test,
                caption=f"Style: {sample['style_text'][0]}, Gen: {sample['gen_text'][0]}{rank_info}"
            )
        
        # Evaluation visualization with style text (only on main process)
        eval_batch = eval_results['last_batch']
        embeddings = eval_results['last_embeddings']
        eval_model_out = eval_results['last_model_out']
        
        if eval_batch is not None and embeddings is not None and eval_model_out is not None:
            eval_model_latent = eval_model_out[1]
            
            if hasattr(self.model, 'module'):
                eval_output = torch.clamp(self.model.module.module_vae_decode(eval_model_latent).sample, 0, 1)
                eval_gt_latent = rearrange(embeddings['decoder_inputs_embeds'], 'b w (h c) -> b c h w',
                                         b=eval_model_latent.shape[0], h=8, c=1)
                eval_input_img = torch.clamp(self.model.module.module_vae_decode(eval_gt_latent).sample, 0, 1)
                real_gen_test = self.model.module.module_continue_gen_test(eval_input_img, {**eval_batch, **embeddings})
            else:
                eval_output = torch.clamp(self.model.vae.decode(eval_model_latent).sample, 0, 1)
                eval_gt_latent = rearrange(embeddings['decoder_inputs_embeds'], 'b w (h c) -> b c h w',
                                         b=eval_model_latent.shape[0], h=8, c=1)
                eval_input_img = torch.clamp(self.model.vae.decode(eval_gt_latent).sample, 0, 1)
                real_gen_test = self.model.continue_gen_test(eval_input_img, {**eval_batch, **embeddings})
            
            eval_out = torch.cat([
                self.processor.pad_images(eval_batch['style_img']).cuda(),
                self.processor.pad_images(eval_batch['gen_img']).cuda(),
                eval_output.repeat(1, 3, 1, 1)
            ], dim=-1)
            
            wandb_data['real_img_with_style'] = wandb.Image(
                make_grid(eval_out, nrow=1, normalize=True),
                caption=f"WITH STYLE - Style: {'|'.join(flatten(eval_batch['style_text']))}, Gen: {'|'.join(flatten(eval_batch['gen_text']))}"
            )
            wandb_data['real_gen_test_with_style'] = wandb.Image(
                real_gen_test,
                caption=f"WITH STYLE - Style: {eval_batch['style_text'][0]}, Gen: {eval_batch['gen_text'][0]}"
            )
        
        # Evaluation visualization without style text (only on main process)
        eval_batch_no_style = eval_results['last_batch_no_style']
        embeddings_no_style = eval_results['last_embeddings_no_style']
        eval_model_out_no_style = eval_results['last_model_out_no_style']
        
        if eval_batch_no_style is not None and embeddings_no_style is not None and eval_model_out_no_style is not None:
            eval_model_latent_no_style = eval_model_out_no_style[1]
            
            if hasattr(self.model, 'module'):
                eval_output_no_style = torch.clamp(self.model.module.module_vae_decode(eval_model_latent_no_style).sample, 0, 1)
                eval_gt_latent_no_style = rearrange(embeddings_no_style['decoder_inputs_embeds'], 'b w (h c) -> b c h w',
                                                  b=eval_model_latent_no_style.shape[0], h=8, c=1)
                eval_input_img_no_style = torch.clamp(self.model.module.module_vae_decode(eval_gt_latent_no_style).sample, 0, 1)
                real_gen_test_no_style = self.model.module.module_continue_gen_test(eval_input_img_no_style, {**eval_batch_no_style, **embeddings_no_style})
            else:
                eval_output_no_style = torch.clamp(self.model.vae.decode(eval_model_latent_no_style).sample, 0, 1)
                eval_gt_latent_no_style = rearrange(embeddings_no_style['decoder_inputs_embeds'], 'b w (h c) -> b c h w',
                                                  b=eval_model_latent_no_style.shape[0], h=8, c=1)
                eval_input_img_no_style = torch.clamp(self.model.vae.decode(eval_gt_latent_no_style).sample, 0, 1)
                real_gen_test_no_style = self.model.continue_gen_test(eval_input_img_no_style, {**eval_batch_no_style, **embeddings_no_style})
            
            eval_out_no_style = torch.cat([
                self.processor.pad_images(eval_batch_no_style['style_img']).cuda(),
                self.processor.pad_images(eval_batch_no_style['gen_img']).cuda(),
                eval_output_no_style.repeat(1, 3, 1, 1)
            ], dim=-1)
            
            wandb_data['real_img_without_style'] = wandb.Image(
                make_grid(eval_out_no_style, nrow=1, normalize=True),
                caption=f"WITHOUT STYLE - Style: {'|'.join(flatten(eval_batch_no_style['style_text']))}, Gen: {'|'.join(flatten(eval_batch_no_style['gen_text']))}"
            )
            wandb_data['real_gen_test_without_style'] = wandb.Image(
                real_gen_test_no_style,
                caption=f"WITHOUT STYLE - Style: {eval_batch_no_style['style_text'][0]}, Gen: {eval_batch_no_style['gen_text'][0]}"
            )
        
        return wandb_data
    
    def save_checkpoint(self, epoch, output_dir, wandb_id):
        """Save model checkpoint"""
        if hasattr(self.model, 'module'):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            'model': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'wandb_id': wandb_id,
            'global_step': self.global_step,
            'samples_processed': self.samples_processed
        }
        
        checkpoint_path = Path(output_dir) / f'{self.global_step:09d}.pth'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.accelerator.is_main_process:
            try:
                temp_path = checkpoint_path.with_suffix('.tmp')
                torch.save(checkpoint, temp_path)
                temp_path.rename(checkpoint_path)
                print(f'Saved checkpoint at epoch {epoch}: {checkpoint_path}')
            except Exception as e:
                print(f'Error saving checkpoint: {e}')
                if temp_path.exists():
                    temp_path.unlink()
    
    def train(self, start_epoch=0, wandb_id=None, run_name=None):
        """Main training loop"""
        # Setup run name and wandb
        if run_name is None:
            if hasattr(self.model, 'module'):
                size = self.model.module.t5_name_or_path.split("-")[2]
            else:
                size = self.model.t5_name_or_path.split("-")[2]
            run_name = f"Eruku_continuous_ce_{size}_bs{self.config.BATCH_SIZE}_multigpu_bigd"
        output_dir = f'{self.config.CHECKPOINT_DIR}/{run_name}'
        
        # Calculate logging frequency
        if self.accelerator.is_main_process:
            samples_per_iteration = self.config.BATCH_SIZE * self.accelerator.num_processes
            effective_batch_size = samples_per_iteration * self.accelerator.gradient_accumulation_steps
            print(f"Logging and checkpointing every {self.config.LOG_CHECKPOINT_SAMPLES} samples.")
            print(f"Dataset sharding: Each GPU processes different data (nodesplitter=wds.split_by_node)")
            print(f"Configured batch size per GPU: {self.config.BATCH_SIZE}")
            print(f"Number of GPUs: {self.accelerator.num_processes}")
            print(f"Gradient accumulation steps: {self.accelerator.gradient_accumulation_steps}")
            print(f"Samples per iteration: {samples_per_iteration}")
            print(f"Effective batch size (per optimizer step): {effective_batch_size}")
            print(f"Expected iterations for first log: {self.config.LOG_CHECKPOINT_SAMPLES // samples_per_iteration}")

        self.setup_wandb(run_name, wandb_id)
        
        if self.accelerator.num_processes > 1:
            print(f"Using distributed training with {self.accelerator.num_processes} processes")
        
        for epoch in range(start_epoch, self.config.MAX_EPOCHS):
            self.model.train()
            # Handle WebDataset which doesn't have len()
            try:
                dataset_len = len(self.train_loader)
                initial_pos = self.global_step % dataset_len
            except TypeError:
                # WebDataset doesn't have __len__, so we can't calculate initial position
                initial_pos = 0
            
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}', disable=not self.accelerator.is_main_process, initial=initial_pos)
            
            # Data verification logging at start of each epoch
            batch_count = 0
            
            for i, sample in enumerate(progress_bar):
                with self.accelerator.accumulate(self.model):
                    # Process batch
                    sample = self.process_batch(sample)
                    
                    # Log first few batches for data distribution verification
                    if batch_count < 3 and epoch == start_epoch:
                        actual_batch_size = len(sample['style_text'])
                        style_img_shape = sample['style_img'].shape
                        gen_img_shape = sample['gen_img'].shape
                        print(f"[GPU {self.accelerator.process_index}] Epoch {epoch} Batch {batch_count}: "
                              f"actual_batch_size={actual_batch_size}, "
                              f"style_img_shape={style_img_shape}, gen_img_shape={gen_img_shape}, "
                              f"style_sample='{sample['style_text'][0][:30]}...', "
                              f"gen_sample='{sample['gen_text'][0][:30]}...'")
                    
                    batch_count += 1
                    
                    # Get model inputs and perform training step
                    model_inputs = self.get_model_inputs(sample)
                    model_out, loss, mse_loss, ce_loss, ocr_loss = self.training_step(sample, model_inputs)

                    if self.accelerator.is_main_process:
                        self.running_loss += loss.detach()
                        self.running_mse_loss += mse_loss.detach()
                        self.running_ce_loss += ce_loss.detach()
                        self.running_ocr_loss += ocr_loss.detach()
                        self.steps_since_log += 1
                        
                        # Track style text dropout statistics
                        if 'style_text_dropped' in sample:
                            style_dropped_count = sum(sample['style_text_dropped'])
                            total_samples = len(sample['style_text_dropped'])
                            self.style_dropout_rate = style_dropped_count / total_samples if total_samples > 0 else 0.0
                
                # Count samples on every iteration (dataset sharding means each GPU processes different data)
                actual_batch_size = len(sample['style_text'])
                self.samples_processed += actual_batch_size * self.accelerator.num_processes
                
                # Update progress bar on every iteration
                progress_bar.set_postfix({
                    "global_step": self.global_step,
                    "samples": self.samples_processed
                })
                
                # Check if it's time to log and checkpoint (every 24k samples) - check on every iteration
                if self.samples_processed > 0 and self.samples_processed % self.config.LOG_CHECKPOINT_SAMPLES == 0:
                    if self.accelerator.is_main_process:
                        print(f"\n🚀 LOGGING AND CHECKPOINTING at {self.samples_processed} samples (global_step={self.global_step})")
                    
                    with torch.inference_mode():
                        eval_results = self.evaluation_step()

                        # =============================================================================
                        # FIX: Add a barrier here to prevent other processes from running ahead.
                        self.accelerator.wait_for_everyone()
                        # =============================================================================

                        # Create visualizations (only on main process)
                        if self.accelerator.is_main_process:
                            wandb_data = self.create_visualization_images(sample, model_inputs, model_out, eval_results)
                            
                            # Calculate average training loss for logging
                            avg_train_loss = self.running_loss / self.steps_since_log if self.steps_since_log > 0 else 0.0
                            avg_train_mse_loss = self.running_mse_loss / self.steps_since_log if self.steps_since_log > 0 else 0.0
                            avg_train_ce_loss = self.running_ce_loss / self.steps_since_log if self.steps_since_log > 0 else 0.0
                            avg_train_ocr_loss = self.running_ocr_loss / self.steps_since_log if self.steps_since_log > 0 else 0.0

                            # Log to wandb with global_step as the step
                            wandb.log({
                                'epoch': epoch,
                                'samples_processed': self.samples_processed,
                                'train/loss': loss, # Log loss from the current step
                                'train/avg_loss': avg_train_loss,
                                'train/mse_loss': avg_train_mse_loss,
                                'train/ce_loss': avg_train_ce_loss,
                                'train/ocr_loss': avg_train_ocr_loss,
                                'train/style_dropout_rate': self.style_dropout_rate,
                                'validation/total_loss': eval_results['eval_loss'],
                                'validation/mse_loss': eval_results['eval_mse_loss'],
                                'validation/ce_loss': eval_results['eval_ce_loss'],
                                'validation/ocr_loss': eval_results['eval_ocr_loss'],
                                'validation/total_loss_no_style': eval_results['eval_loss_no_style'],
                                'validation/mse_loss_no_style': eval_results['eval_mse_loss_no_style'],
                                'validation/ce_loss_no_style': eval_results['eval_ce_loss_no_style'],
                                'validation/ocr_loss': eval_results['eval_ocr_loss'],
                                'validation/ocr_loss_no_style': eval_results['eval_ocr_loss_no_style'],
                                **wandb_data
                            }, step=self.global_step)
                            del wandb_data

                            # Reset running loss accumulator
                            self.running_loss = 0.0
                            self.running_mse_loss = 0.0
                            self.running_ce_loss = 0.0
                            self.running_ocr_loss = 0.0
                            self.steps_since_log = 0
                    
                    # Save checkpoint
                    self.save_checkpoint(epoch, output_dir, wandb.run.id if self.accelerator.is_main_process else None)
                
                # Increment global step only after a real optimization step
                if self.accelerator.sync_gradients:
                    self.global_step += 1

            # Optional: a final evaluation at the end of each epoch regardless of step
            self.accelerator.wait_for_everyone()
            with torch.inference_mode():
                final_eval = self.evaluation_step()
                if self.accelerator.is_main_process:
                    wandb.log({
                        'epoch': epoch,
                        'samples_processed': self.samples_processed,
                        'eval/epoch_end_loss': final_eval['eval_loss'],
                        'eval/epoch_end_loss_no_style': final_eval['eval_loss_no_style'],
                        'eval/epoch_end_ocr_loss': final_eval['eval_ocr_loss'],
                        'eval/epoch_end_ocr_loss_no_style': final_eval['eval_ocr_loss_no_style']
                    }, step=self.global_step)
                    print(f"Epoch {epoch} finished. Eval loss: {final_eval['eval_loss']:.4f}, Eval loss (no style): {final_eval['eval_loss_no_style']:.4f}, Eval OCR loss: {final_eval['eval_ocr_loss']:.4f}, Eval OCR loss (no style): {final_eval['eval_ocr_loss_no_style']:.4f}, Total samples processed: {self.samples_processed}")


class SHTGWrapper(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.transforms = T.Compose([
            self._to_height_64,
            T.ToTensor()
        ])

    def _to_height_64(self, img):
        width, height = img.size
        aspect_ratio = width / height
        new_width = int(64 * aspect_ratio)
        resized_image = img.resize((new_width, 64), Image.LANCZOS)
        return resized_image

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        sample['style_img'] = self.transforms(sample['style_imgs'][0].convert('RGB'))
        return sample


def flatten(l):
    # Flattens a list of lists or returns the list if already flat
    if l and isinstance(l[0], list):
        return [item for sublist in l for item in sublist]
    return l


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Multi-GPU Training Script for Eruku Continuous")
    
    parser.add_argument(
        "--run_name", 
        type=str, 
        default=None,
        help="Custom run name for wandb and checkpoint directory (default: auto-generated)"
    )
    
    parser.add_argument(
        "--override_checkpoint_run_name",
        type=str,
        default=None,
        help="Override checkpoint run name to resume from a specific run"
    )
    
    parser.add_argument(
        "--style_text_dropout_prob",
        type=float,
        default=0,
        help="Probability of replacing style text with empty string during training (default: 0.1)"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Alpha value for the loss function (default: 1.0)"
    )

    parser.add_argument(
        "--with-cfg",
        action="store_true",
        help="Enable CFG"
    )

    parser.add_argument(
        "--ft-dataset",
        action="store_true",
        help="Use the ft dataset"
    )

    parser.add_argument(
        "--override-batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )

    parser.add_argument(
        "--override-max-img-len",
        type=int,
        default=None,
        help="Override max image length"
    )
    
    return parser.parse_args()


def main():
    """Main function to run training"""
    args = parse_args()
    config = Config()
    
    # Override config with CLI arguments
    config.STYLE_TEXT_DROPOUT_PROB = args.style_text_dropout_prob
    config.PERFORM_CFG = args.with_cfg
    if args.override_batch_size is not None:
        config.BATCH_SIZE = args.override_batch_size
    if args.override_max_img_len is not None:
        config.MAX_IMG_LEN = args.override_max_img_len
    if args.ft_dataset:
        config.TRAIN_PATTERN = "/scratch/project_465002151/font-square-pairs-ft/{000000..000988}.tar"


    # Initialize training manager
    trainer = TrainingManager(config)
    trainer.setup_model_and_optimizer()
    trainer.setup_data_loaders()
    trainer.prepare_for_training()
    
    # Generate run name if not provided
    if args.run_name is None:
        if hasattr(trainer.model, 'module'):
            size = trainer.model.module.t5_name_or_path.split("-")[2]
        else:
            size = trainer.model.t5_name_or_path.split("-")[2]
        run_name = f"Eruku_continuous_ce_{size}_bs{config.BATCH_SIZE}_multigpu_bigd"
    else:
        run_name = args.run_name
    
    output_dir = f'{config.CHECKPOINT_DIR}/{run_name}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint if available
    start_epoch, wandb_id = trainer.load_checkpoint(output_dir, args.override_checkpoint_run_name)
    
    if args.alpha != 1.0:
        trainer.model.module.alpha = args.alpha

    # Start training with custom run name
    trainer.train(start_epoch, wandb_id, run_name=run_name)


if __name__ == "__main__":
    main()