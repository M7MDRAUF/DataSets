"""
Batch Processor for CineMatch V2.1.6

Implements batch processing for API requests:
- Request batching
- Parallel processing
- Progress tracking
- Error handling

Phase 2 - Task 2.3: Batch Processing
"""

import logging
import time
import asyncio
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import threading
import queue

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class BatchStatus(Enum):
    """Batch processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some items failed


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 100
    max_workers: int = 4
    timeout_seconds: float = 60.0
    retry_failed: bool = True
    max_retries: int = 3
    continue_on_error: bool = True


@dataclass
class BatchResult(Generic[R]):
    """Result of a batch operation."""
    batch_id: str
    status: BatchStatus
    total_items: int
    successful: int
    failed: int
    results: List[R]
    errors: List[Dict[str, Any]]
    processing_time: float
    
    @property
    def success_rate(self) -> float:
        return self.successful / max(1, self.total_items)


@dataclass
class BatchProgress:
    """Progress of batch processing."""
    total: int
    completed: int
    failed: int
    current_batch: int
    total_batches: int
    elapsed_seconds: float
    
    @property
    def percent_complete(self) -> float:
        return (self.completed + self.failed) / max(1, self.total) * 100
    
    @property
    def remaining(self) -> int:
        return self.total - self.completed - self.failed


class BatchProcessor(Generic[T, R]):
    """
    Batch processor for handling large numbers of items.
    
    Features:
    - Automatic batching
    - Parallel processing
    - Progress callbacks
    - Error handling and retry
    """
    
    def __init__(
        self,
        processor: Callable[[T], R],
        config: Optional[BatchConfig] = None
    ):
        self.processor = processor
        self.config = config or BatchConfig()
        self._batch_counter = 0
        self._lock = threading.Lock()
    
    def _generate_batch_id(self) -> str:
        with self._lock:
            self._batch_counter += 1
            return f"batch_{self._batch_counter:06d}"
    
    def process(
        self,
        items: List[T],
        progress_callback: Optional[Callable[[BatchProgress], None]] = None
    ) -> BatchResult[R]:
        """
        Process items in batches.
        
        Args:
            items: Items to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            BatchResult with results
        """
        batch_id = self._generate_batch_id()
        start_time = time.time()
        
        total = len(items)
        results: List[R] = []
        errors: List[Dict[str, Any]] = []
        completed = 0
        failed = 0
        
        # Split into batches
        batches = [
            items[i:i + self.config.batch_size]
            for i in range(0, len(items), self.config.batch_size)
        ]
        
        logger.info(f"Processing {total} items in {len(batches)} batches")
        
        for batch_idx, batch in enumerate(batches):
            batch_results, batch_errors = self._process_batch(
                batch,
                batch_idx
            )
            
            results.extend(batch_results)
            errors.extend(batch_errors)
            completed += len(batch_results)
            failed += len(batch_errors)
            
            if progress_callback:
                progress = BatchProgress(
                    total=total,
                    completed=completed,
                    failed=failed,
                    current_batch=batch_idx + 1,
                    total_batches=len(batches),
                    elapsed_seconds=time.time() - start_time
                )
                progress_callback(progress)
        
        processing_time = time.time() - start_time
        
        # Determine status
        if failed == 0:
            status = BatchStatus.COMPLETED
        elif completed == 0:
            status = BatchStatus.FAILED
        else:
            status = BatchStatus.PARTIAL
        
        logger.info(
            f"Batch {batch_id}: {completed}/{total} successful "
            f"in {processing_time:.2f}s"
        )
        
        return BatchResult(
            batch_id=batch_id,
            status=status,
            total_items=total,
            successful=completed,
            failed=failed,
            results=results,
            errors=errors,
            processing_time=processing_time
        )
    
    def _process_batch(
        self,
        batch: List[T],
        batch_idx: int
    ) -> tuple[List[R], List[Dict[str, Any]]]:
        """Process a single batch in parallel."""
        results: List[R] = []
        errors: List[Dict[str, Any]] = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_item = {
                executor.submit(self._process_item, item, idx): (idx, item)
                for idx, item in enumerate(batch)
            }
            
            for future in as_completed(
                future_to_item,
                timeout=self.config.timeout_seconds
            ):
                idx, item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    if self.config.continue_on_error:
                        errors.append({
                            'index': batch_idx * self.config.batch_size + idx,
                            'item': str(item)[:100],
                            'error': str(e),
                            'error_type': type(e).__name__
                        })
                    else:
                        raise
        
        return results, errors
    
    def _process_item(self, item: T, idx: int) -> R:
        """Process a single item with retry."""
        last_error: Optional[Exception] = None
        
        for attempt in range(self.config.max_retries):
            try:
                return self.processor(item)
            except Exception as e:
                last_error = e
                if not self.config.retry_failed or attempt == self.config.max_retries - 1:
                    raise
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
        
        raise last_error or RuntimeError("Unexpected error")


class AsyncBatchProcessor(Generic[T, R]):
    """
    Async batch processor for handling large numbers of items.
    
    Features:
    - Async processing
    - Semaphore-based concurrency control
    - Progress tracking
    """
    
    def __init__(
        self,
        processor: Callable[[T], Any],  # Can be sync or async
        config: Optional[BatchConfig] = None
    ):
        self.processor = processor
        self.config = config or BatchConfig()
        self._batch_counter = 0
    
    async def process(
        self,
        items: List[T],
        progress_callback: Optional[Callable[[BatchProgress], None]] = None
    ) -> BatchResult[R]:
        """Process items asynchronously."""
        batch_id = f"async_batch_{self._batch_counter:06d}"
        self._batch_counter += 1
        start_time = time.time()
        
        total = len(items)
        results: List[R] = []
        errors: List[Dict[str, Any]] = []
        
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def process_with_semaphore(idx: int, item: T):
            async with semaphore:
                try:
                    if asyncio.iscoroutinefunction(self.processor):
                        result = await self.processor(item)
                    else:
                        result = self.processor(item)
                    return idx, result, None
                except Exception as e:
                    return idx, None, str(e)
        
        # Process all items
        tasks = [
            process_with_semaphore(idx, item)
            for idx, item in enumerate(items)
        ]
        
        completed = 0
        for coro in asyncio.as_completed(tasks):
            idx, result, error = await coro
            
            if error:
                errors.append({
                    'index': idx,
                    'error': error
                })
            else:
                results.append(result)
            
            completed += 1
            
            if progress_callback:
                progress = BatchProgress(
                    total=total,
                    completed=len(results),
                    failed=len(errors),
                    current_batch=1,
                    total_batches=1,
                    elapsed_seconds=time.time() - start_time
                )
                progress_callback(progress)
        
        processing_time = time.time() - start_time
        
        return BatchResult(
            batch_id=batch_id,
            status=BatchStatus.COMPLETED if not errors else BatchStatus.PARTIAL,
            total_items=total,
            successful=len(results),
            failed=len(errors),
            results=results,
            errors=errors,
            processing_time=processing_time
        )


class BatchQueue(Generic[T]):
    """
    Queue-based batch processor for continuous processing.
    
    Features:
    - Queue-based input
    - Automatic batching
    - Background processing
    """
    
    def __init__(
        self,
        processor: Callable[[List[T]], None],
        batch_size: int = 100,
        flush_interval: float = 1.0
    ):
        self.processor = processor
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        self._queue: queue.Queue[T] = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """Start the batch processor."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info("BatchQueue started")
    
    def stop(self, flush: bool = True) -> None:
        """Stop the batch processor."""
        self._running = False
        
        if flush:
            self._flush()
        
        if self._thread:
            self._thread.join(timeout=5.0)
        
        logger.info("BatchQueue stopped")
    
    def add(self, item: T) -> None:
        """Add an item to the queue."""
        self._queue.put(item)
    
    def add_many(self, items: List[T]) -> None:
        """Add multiple items to the queue."""
        for item in items:
            self._queue.put(item)
    
    def _process_loop(self) -> None:
        """Main processing loop."""
        last_flush = time.time()
        batch: List[T] = []
        
        while self._running:
            try:
                # Get item with timeout
                item = self._queue.get(timeout=0.1)
                batch.append(item)
            except queue.Empty:
                pass
            
            # Check if we should flush
            should_flush = (
                len(batch) >= self.batch_size or
                (batch and time.time() - last_flush >= self.flush_interval)
            )
            
            if should_flush and batch:
                try:
                    self.processor(batch)
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                
                batch = []
                last_flush = time.time()
    
    def _flush(self) -> None:
        """Flush remaining items."""
        batch: List[T] = []
        
        while not self._queue.empty():
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        
        if batch:
            try:
                self.processor(batch)
            except Exception as e:
                logger.error(f"Final batch processing error: {e}")


# Convenience functions
def process_in_batches(
    items: List[T],
    processor: Callable[[T], R],
    batch_size: int = 100,
    max_workers: int = 4
) -> BatchResult[R]:
    """Process items in batches."""
    config = BatchConfig(batch_size=batch_size, max_workers=max_workers)
    batch_processor: BatchProcessor[T, R] = BatchProcessor(processor, config)
    return batch_processor.process(items)


if __name__ == "__main__":
    import random
    
    print("Batch Processor Demo")
    print("=" * 50)
    
    # Sample processor
    def process_item(x: int) -> int:
        time.sleep(0.01)  # Simulate work
        if random.random() < 0.05:  # 5% failure rate
            raise ValueError(f"Random failure for {x}")
        return x * 2
    
    # Create processor
    processor: BatchProcessor[int, int] = BatchProcessor(
        process_item,
        BatchConfig(batch_size=20, max_workers=4)
    )
    
    # Process items
    items = list(range(100))
    
    def show_progress(p: BatchProgress):
        print(f"\rProgress: {p.percent_complete:.1f}% ({p.completed}/{p.total})", end="")
    
    print("\nProcessing 100 items...")
    result = processor.process(items, progress_callback=show_progress)
    
    print(f"\n\nResults:")
    print(f"  Status: {result.status.value}")
    print(f"  Successful: {result.successful}")
    print(f"  Failed: {result.failed}")
    print(f"  Success Rate: {result.success_rate:.1%}")
    print(f"  Processing Time: {result.processing_time:.2f}s")
    
    if result.errors:
        print(f"\n  First few errors:")
        for err in result.errors[:3]:
            print(f"    - Item {err['index']}: {err['error']}")
