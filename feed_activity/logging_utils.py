"""실행 중 관측치 기록: CSV 로그 저장 + 구간별(read/track/feature/draw/encode_write) 처리시간 프로파일링"""
import collections
import csv
import os
import time
from contextlib import contextmanager
from datetime import datetime

CSV_HEADER = [
    "index",
    "time_sec", "flow_intensity", "fg_pixel_ratio",
    "total_mv_px", "avg_mv_px_per_object_sec", "object_count",
    "feeding_score_pct", "feeding_score_raw_pct",
]


class FeedingCsvLogger:
    def __init__(self, csv_dir: str):
        os.makedirs(csv_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(csv_dir, f"feeding_log_{ts}.csv")
        self._file = open(self.path, "w", newline="", encoding="utf-8-sig")
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_HEADER)
        self._row_index = 0  # 3초(--sample-sec)마다 한 줄씩 쌓일 때 1부터 매기는 일련번호
        print(f"[CSV] 저장 경로: {self.path}")

    def write_row(self, elapsed, feats, total_mv_px, avg_mv_px_per_object_sec,
                  object_count, score_pct, raw_score_pct=None):
        """score_pct: 급락 방지(clamp)까지 적용된 최종 발표용 점수.
        raw_score_pct: clamp 적용 전 원본 점수(생략하면 score_pct와 동일하게 기록 —
        기존 호출부와의 하위 호환용). 두 값을 나란히 남겨두면 clamp가 실제로
        언제/얼마나 개입했는지 CSV만 보고도 확인할 수 있다."""
        self._row_index += 1
        if raw_score_pct is None:
            raw_score_pct = score_pct
        self._writer.writerow([
            self._row_index,
            f"{elapsed:.2f}",
            f"{feats['flow_intensity']:.4f}", f"{feats['fg_pixel_ratio']:.4f}",
            f"{total_mv_px:.2f}", f"{avg_mv_px_per_object_sec:.2f}", object_count,
            f"{score_pct:.2f}", f"{raw_score_pct:.2f}",
        ])
        self._file.flush()

    def close(self):
        self._file.close()
        print(f"CSV 저장 완료: {self.path}")


class StageProfiler:
    """with profiler.measure("read"): ... 형태로 감싼 구간의 누적 시간을,
    print_every 프레임마다 frame당 평균(ms)으로 출력"""

    def __init__(self, enabled: bool = True, print_every: int = 60):
        self.enabled = enabled
        self.print_every = print_every
        self._stage_times = collections.defaultdict(float)
        self._count = 0

    @contextmanager
    def measure(self, stage_name: str):
        if not self.enabled:
            yield
            return
        t0 = time.monotonic()
        try:
            yield
        finally:
            self._stage_times[stage_name] += time.monotonic() - t0

    def tick(self):
        """프레임 1개 처리가 끝날 때마다 호출, print_every에 도달하면 결과를 출력"""
        if not self.enabled:
            return
        self._count += 1
        if self._count >= self.print_every:
            total_ms = sum(self._stage_times.values()) / self._count * 1000
            parts = ", ".join(
                f"{k}={v / self._count * 1000:.1f}ms" for k, v in self._stage_times.items()
            )
            print(f"[PROFILE] frame당 합계 {total_ms:.1f}ms ({parts})")
            self._stage_times.clear()
            self._count = 0
