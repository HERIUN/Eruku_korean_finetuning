"""영어 pretrained 에서 바로 한국어(긴 줄)를 학습 — Phase 1 생략.

목적: 영어 pretrained 의 style fidelity(영어를 또렷·in-style·전체문장 생성)를 계승하면서
      한국어를 얹는다. Phase 1(짧은어절 glyph 집중학습)을 거치지 않아 영어 능력 손상을 줄임.

  python train_korean_from_en.py
  CUDA_VISIBLE_DEVICES=2 python train_korean_from_en.py

설계 (사용자 확정):
- 시작: 영어 pretrained (--resume 없음 → eruku_pretrained 자동 로드/다운로드)
- 데이터: 한0.45/영0.2/숫자0.2/랜덤0.15 (현행 유지, 영어 forgetting 방지용으로 영어 계속 노출)
- 어절: 긴 줄 (style 1~8 / gen 1~32)
- lr 5e-5 (1e-4 보다 낮춰 pretrained 천천히 덮음 = 영어 보존)
- 적용된 수정: max-img-len 8192(긴 gen 안 잘림), warp p=0.5(휨 완화), width 분배(원본식)

산출물: finetune_runs/korean_en2ko/
"""
from train_core import make_parser, train


def main():
    p = make_parser()
    p.set_defaults(
        online_split=True,
        style_words=[1, 8], gen_words=[1, 32],
        text_dropout=0.05, style_text_dropout=0.10,
        batch_size=16, grad_accum=16, lr=5e-5,       # virtual batch 256 = 16 x 16 (측정: 8192 에서 VRAM 35GB)
        num_workers=16,
        max_img_len=8192,
        max_steps=20000, save_every=1000, log_every=10,
        save_samples=16,
        val_every=1000, val_batches=4, val_seed=1234,  # held-out(unseen 16폰트) val loss 로깅
        english_frac=0.15,                            # 15% 영어 전용(라틴 177폰트) = 영어 스타일 다양성
        resume=None,                                  # 영어 pretrained 에서 시작 (Phase1 생략)
        out="finetune_runs/korean_en2ko",
    )
    train(p.parse_args())


if __name__ == "__main__":
    main()
