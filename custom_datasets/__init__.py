"""데이터셋 패키지 — **upstream(원본) 과 korean(이 repo 추가분) 을 물리적으로 분리**한다.

    custom_datasets/
    ├─ upstream/   Eruku 원본 vendored 복사본. 수정 금지(부득이한 수정은 upstream/README.md 에 기록)
    └─ korean/     이 저장소가 추가한 한글 데이터 경로. upstream 을 import 해서 확장한다

의존 방향은 **korean → upstream 단방향**이다. upstream 코드가 korean 을 import 하면 안 된다
(원본과 diff 를 뜰 수 없게 된다).

아래 re-export 는 하위호환용이다. 새 코드는 `custom_datasets.upstream...` /
`custom_datasets.korean...` 를 명시적으로 import 한다.
"""
from .upstream import *                                  # noqa: F401,F403
from .upstream.alphabet import Alphabet                   # noqa: F401
from .upstream.real_datasets.datasets import dataset_factory   # noqa: F401
