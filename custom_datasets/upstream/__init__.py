"""Eruku 원본(upstream) 데이터 코드의 vendored 복사본.

출처: <https://github.com/Blowing-Up-Groundhogs/Eruku>
**이 디렉토리는 원본과 diff 를 뜰 수 있게 유지한다.** 한글용 변경은 여기가 아니라
`custom_datasets/korean/` 에 서브클래스·래퍼로 넣는다. 부득이하게 원본을 고쳤다면
[`README.md`](README.md) 의 표에 반드시 기록한다.
"""
from .font_square import *                        # noqa: F401,F403
from .alphabet import Alphabet                    # noqa: F401
from .real_datasets.datasets import dataset_factory   # noqa: F401
