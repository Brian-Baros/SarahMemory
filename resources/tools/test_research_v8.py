"""
SarahMemoryResearch.py v8.0.0 - Comprehensive Test Suite

This test suite validates all features of the upgraded research engine.

Usage:
    python test_research_v8.py
    
Requirements:
    - pytest (optional, but recommended)
    - All SarahMemory dependencies
"""

import sys
import os
import time
import asyncio
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the upgraded research module
from SarahMemoryResearch import (
    get_research_data,
    parallel_research,
    LocalResearch,
    WebResearch,
    APIResearch,
    ResearchResult,
    ResearchSource,
    get_cached_result,
    cache_result,
    clear_cache,
    get_research_metrics,
    reset_research_metrics,
    STATIC_FACTS
)

# ============================================================================
# TEST UTILITIES
# ============================================================================

class TestReport:
    """Test result tracking."""
    def __init__(self):
        self.passed = []
        self.failed = []
        self.skipped = []
        self.start_time = time.time()
    
    def record_pass(self, test_name: str):
        self.passed.append(test_name)
        print(f"✓ PASS: {test_name}")
    
    def record_fail(self, test_name: str, error: str):
        self.failed.append((test_name, error))
        print(f"✗ FAIL: {test_name}")
        print(f"  Error: {error}")
    
    def record_skip(self, test_name: str, reason: str):
        self.skipped.append((test_name, reason))
        print(f"⊘ SKIP: {test_name} ({reason})")
    
    def print_summary(self):
        elapsed = time.time() - self.start_time
        total = len(self.passed) + len(self.failed) + len(self.skipped)
        
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Total Tests:  {total}")
        print(f"Passed:       {len(self.passed)} ({len(self.passed)/max(total,1)*100:.1f}%)")
        print(f"Failed:       {len(self.failed)} ({len(self.failed)/max(total,1)*100:.1f}%)")
        print(f"Skipped:      {len(self.skipped)} ({len(self.skipped)/max(total,1)*100:.1f}%)")
        print(f"Time:         {elapsed:.2f}s")
        print("=" * 70)
        
        if self.failed:
            print("\nFAILED TESTS:")
            for test_name, error in self.failed:
                print(f"  - {test_name}: {error}")
        
        return len(self.failed) == 0


report = TestReport()


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================

def test_basic_query():
    """Test basic query execution."""
    try:
        result = get_research_data("what is pi")
        
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'data' in result, "Result should have 'data' key"
        assert 'source' in result, "Result should have 'source' key"
        assert 'confidence' in result, "Result should have 'confidence' key"
        assert isinstance(result['confidence'], (int, float)), "Confidence should be numeric"
        assert 0.0 <= result['confidence'] <= 1.0, "Confidence should be 0-1"
        
        report.record_pass("Basic Query Execution")
    except Exception as e:
        report.record_fail("Basic Query Execution", str(e))


def test_static_fact_lookup():
    """Test static fact retrieval."""
    try:
        result = get_research_data("what is the speed of light")
        
        assert result['confidence'] == 1.0, "Static facts should have 100% confidence"
        assert 'local_static' in result['source'], "Should identify as static source"
        assert '299,792,458' in result['data'] or '299792458' in result['data'], \
            "Should contain speed of light value"
        
        report.record_pass("Static Fact Lookup")
    except Exception as e:
        report.record_fail("Static Fact Lookup", str(e))


def test_async_parallel_research():
    """Test async parallel research."""
    try:
        async def run_test():
            result = await parallel_research("test query")
            assert isinstance(result, ResearchResult), "Should return ResearchResult object"
            assert hasattr(result, 'source'), "Should have source attribute"
            assert hasattr(result, 'content'), "Should have content attribute"
            assert hasattr(result, 'confidence'), "Should have confidence attribute"
            return True
        
        success = asyncio.run(run_test())
        assert success, "Async execution failed"
        
        report.record_pass("Async Parallel Research")
    except Exception as e:
        report.record_fail("Async Parallel Research", str(e))


# ============================================================================
# CACHING TESTS
# ============================================================================

def test_cache_functionality():
    """Test result caching."""
    try:
        clear_cache()
        
        # First query (cache miss)
        start1 = time.time()
        result1 = get_research_data("test cache query")
        time1 = time.time() - start1
        
        # Second identical query (cache hit)
        start2 = time.time()
        result2 = get_research_data("test cache query")
        time2 = time.time() - start2
        
        assert result1['data'] == result2['data'], "Cached result should match original"
        assert time2 < time1 * 0.1, f"Cache should be much faster (was {time2:.3f}s vs {time1:.3f}s)"
        
        report.record_pass("Cache Functionality")
    except Exception as e:
        report.record_fail("Cache Functionality", str(e))


def test_cache_expiration():
    """Test cache TTL expiration."""
    try:
        clear_cache()
        
        # Create a test result
        test_result = ResearchResult(
            source=ResearchSource.LOCAL_STATIC,
            content="test content",
            query="expiration test"
        )
        
        # Cache with short TTL
        cache_result("expiration test", test_result, ttl=1)
        
        # Immediate retrieval should work
        cached = get_cached_result("expiration test")
        assert cached is not None, "Should retrieve immediately after caching"
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should be expired now
        cached = get_cached_result("expiration test")
        assert cached is None, "Should expire after TTL"
        
        report.record_pass("Cache Expiration")
    except Exception as e:
        report.record_fail("Cache Expiration", str(e))


# ============================================================================
# SOURCE-SPECIFIC TESTS
# ============================================================================

def test_local_search():
    """Test local database search."""
    try:
        # Test with a query that should hit static facts
        result = LocalResearch.search("what is pi", "question")
        
        assert result is not None, "Should find result in local sources"
        assert isinstance(result, ResearchResult), "Should return ResearchResult"
        assert result.source == ResearchSource.LOCAL_STATIC, "Should identify correct source"
        
        report.record_pass("Local Search")
    except Exception as e:
        report.record_fail("Local Search", str(e))


def test_web_preprocessing():
    """Test query preprocessing."""
    try:
        preprocessed = WebResearch.preprocess_query("What is the capital of France?")
        
        assert "what is" not in preprocessed.lower(), "Should remove 'what is' prefix"
        assert "?" not in preprocessed, "Should remove question mark"
        assert "capital of France" in preprocessed, "Should preserve main query content"
        
        report.record_pass("Web Query Preprocessing")
    except Exception as e:
        report.record_fail("Web Query Preprocessing", str(e))


def test_captcha_detection():
    """Test DuckDuckGo CAPTCHA detection."""
    try:
        # Test with known CAPTCHA indicators
        captcha_html = "Please complete the following challenge to verify you're human"
        is_blocked = WebResearch._is_ddg_blocked(captcha_html)
        assert is_blocked, "Should detect CAPTCHA challenge"
        
        # Test with normal content
        normal_html = "<html><body>Normal search results</body></html>"
        is_blocked = WebResearch._is_ddg_blocked(normal_html)
        assert not is_blocked, "Should not flag normal content"
        
        report.record_pass("CAPTCHA Detection")
    except Exception as e:
        report.record_fail("CAPTCHA Detection", str(e))


# ============================================================================
# CONFIDENCE SCORING TESTS
# ============================================================================

def test_confidence_scores():
    """Test confidence scoring accuracy."""
    try:
        # Static facts should have highest confidence
        result1 = get_research_data("what is the speed of light")
        assert result1['confidence'] == 1.0, "Static facts should have 1.0 confidence"
        
        # Any result should have valid confidence
        result2 = get_research_data("random test query xyz123")
        assert 0.0 <= result2['confidence'] <= 1.0, "Confidence must be 0-1"
        
        report.record_pass("Confidence Scoring")
    except Exception as e:
        report.record_fail("Confidence Scoring", str(e))


# ============================================================================
# METRICS & MONITORING TESTS
# ============================================================================

def test_metrics_tracking():
    """Test performance metrics tracking."""
    try:
        reset_research_metrics()
        
        # Execute some queries
        get_research_data("test query 1")
        get_research_data("test query 2")
        get_research_data("test query 1")  # Cache hit
        
        stats = get_research_metrics()
        
        assert stats['total_queries'] >= 2, "Should track total queries"
        assert stats['cache_hits'] >= 1, "Should track cache hits"
        assert 'avg_latency_ms' in stats, "Should track latency"
        assert stats['avg_latency_ms'] >= 0, "Latency should be non-negative"
        
        report.record_pass("Metrics Tracking")
    except Exception as e:
        report.record_fail("Metrics Tracking", str(e))


# ============================================================================
# BACKWARD COMPATIBILITY TESTS
# ============================================================================

def test_backward_compatibility():
    """Test backward compatibility with v7.7.5 API."""
    try:
        # Old API style (should still work)
        result = get_research_data("test")
        
        # Check old keys still exist
        assert 'data' in result, "Should have 'data' key (v7.7.5 format)"
        assert 'source' in result, "Should have 'source' key (v7.7.5 format)"
        assert 'intent' in result, "Should have 'intent' key (v7.7.5 format)"
        
        # Result should be dictionary (not ResearchResult object)
        assert isinstance(result, dict), "Should return dict for backward compatibility"
        
        report.record_pass("Backward Compatibility")
    except Exception as e:
        report.record_fail("Backward Compatibility", str(e))


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_graceful_degradation():
    """Test graceful failure handling."""
    try:
        # Query with all sources likely to fail
        result = get_research_data("xyzabc123impossible query that wont match anything")
        
        # Should still return a result (not crash)
        assert isinstance(result, dict), "Should return result even on failure"
        assert 'data' in result, "Should have data key"
        assert 'confidence' in result, "Should have confidence key"
        
        # Confidence should be low
        assert result['confidence'] <= 0.5, "Failed query should have low confidence"
        
        report.record_pass("Graceful Degradation")
    except Exception as e:
        report.record_fail("Graceful Degradation", str(e))


def test_empty_query():
    """Test handling of empty/invalid queries."""
    try:
        result = get_research_data("")
        
        assert isinstance(result, dict), "Should handle empty query gracefully"
        assert result['confidence'] <= 0.5, "Empty query should have low confidence"
        
        report.record_pass("Empty Query Handling")
    except Exception as e:
        report.record_fail("Empty Query Handling", str(e))


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

def test_parallel_speedup():
    """Test that parallel execution is faster than sequential."""
    try:
        # Sequential execution simulation
        start_seq = time.time()
        result1 = get_research_data("test query A")
        result2 = get_research_data("test query B")
        time_seq = time.time() - start_seq
        
        # Clear cache for fair comparison
        clear_cache()
        
        # Parallel execution
        async def parallel_test():
            start = time.time()
            results = await asyncio.gather(
                parallel_research("test query A"),
                parallel_research("test query B")
            )
            return time.time() - start
        
        time_parallel = asyncio.run(parallel_test())
        
        # Parallel should be at least somewhat faster (accounting for overhead)
        # We use 0.8 as threshold to account for variability
        assert time_parallel < time_seq * 0.8 or time_seq < 0.5, \
            f"Parallel should be faster (seq: {time_seq:.2f}s, par: {time_parallel:.2f}s)"
        
        report.record_pass("Parallel Speedup")
    except Exception as e:
        report.record_fail("Parallel Speedup", str(e))


def test_cache_speedup():
    """Test cache performance improvement."""
    try:
        clear_cache()
        
        # First query (no cache)
        start1 = time.time()
        get_research_data("cache speedup test query")
        time1 = time.time() - start1
        
        # Second query (cached)
        start2 = time.time()
        get_research_data("cache speedup test query")
        time2 = time.time() - start2
        
        # Cache should be at least 10x faster
        speedup = time1 / max(time2, 0.001)  # Avoid division by zero
        assert speedup > 10, f"Cache should be much faster (speedup: {speedup:.1f}x)"
        
        report.record_pass("Cache Speedup")
    except Exception as e:
        report.record_fail("Cache Speedup", str(e))


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_static_facts_coverage():
    """Test static facts database coverage."""
    try:
        assert len(STATIC_FACTS) >= 20, f"Should have 20+ static facts (has {len(STATIC_FACTS)})"
        
        # Test a few specific facts
        required_facts = [
            "what is pi",
            "what is the speed of light",
            "what is dna"
        ]
        
        for fact_query in required_facts:
            assert fact_query in STATIC_FACTS, f"Missing required fact: {fact_query}"
        
        report.record_pass("Static Facts Coverage")
    except Exception as e:
        report.record_fail("Static Facts Coverage", str(e))


def test_research_result_structure():
    """Test ResearchResult data structure."""
    try:
        result = ResearchResult(
            source=ResearchSource.LOCAL_STATIC,
            content="test content",
            confidence=0.95,
            query="test",
            intent="question",
            latency_ms=100
        )
        
        # Test attributes
        assert result.source == ResearchSource.LOCAL_STATIC
        assert result.content == "test content"
        assert result.confidence == 0.95
        assert result.latency_ms == 100
        
        # Test to_dict conversion
        dict_result = result.to_dict()
        assert isinstance(dict_result, dict)
        assert 'data' in dict_result
        assert 'source' in dict_result
        assert 'confidence' in dict_result
        
        report.record_pass("ResearchResult Structure")
    except Exception as e:
        report.record_fail("ResearchResult Structure", str(e))


# ============================================================================
# STRESS TESTS
# ============================================================================

def test_concurrent_queries():
    """Test handling of concurrent queries."""
    try:
        async def stress_test():
            # Fire 10 concurrent queries
            queries = [f"concurrent test query {i}" for i in range(10)]
            tasks = [parallel_research(q) for q in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should complete (even if some fail)
            assert len(results) == 10, "Should handle all concurrent queries"
            
            # Check for exceptions
            exceptions = [r for r in results if isinstance(r, Exception)]
            assert len(exceptions) == 0, f"Should not raise exceptions ({len(exceptions)} found)"
            
            return True
        
        success = asyncio.run(stress_test())
        assert success
        
        report.record_pass("Concurrent Queries")
    except Exception as e:
        report.record_fail("Concurrent Queries", str(e))


def test_rapid_fire_queries():
    """Test rapid sequential queries."""
    try:
        # Fire 20 queries as fast as possible
        for i in range(20):
            result = get_research_data(f"rapid fire test {i}")
            assert isinstance(result, dict), f"Query {i} failed"
        
        report.record_pass("Rapid Fire Queries")
    except Exception as e:
        report.record_fail("Rapid Fire Queries", str(e))


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all test suites."""
    print("=" * 70)
    print("SarahMemoryResearch.py v8.0.0 - Test Suite")
    print("=" * 70)
    print()
    
    # Basic functionality
    print("BASIC FUNCTIONALITY TESTS")
    print("-" * 70)
    test_basic_query()
    test_static_fact_lookup()
    test_async_parallel_research()
    print()
    
    # Caching
    print("CACHING TESTS")
    print("-" * 70)
    test_cache_functionality()
    test_cache_expiration()
    print()
    
    # Source-specific
    print("SOURCE-SPECIFIC TESTS")
    print("-" * 70)
    test_local_search()
    test_web_preprocessing()
    test_captcha_detection()
    print()
    
    # Confidence scoring
    print("CONFIDENCE SCORING TESTS")
    print("-" * 70)
    test_confidence_scores()
    print()
    
    # Metrics
    print("METRICS & MONITORING TESTS")
    print("-" * 70)
    test_metrics_tracking()
    print()
    
    # Backward compatibility
    print("BACKWARD COMPATIBILITY TESTS")
    print("-" * 70)
    test_backward_compatibility()
    print()
    
    # Error handling
    print("ERROR HANDLING TESTS")
    print("-" * 70)
    test_graceful_degradation()
    test_empty_query()
    print()
    
    # Performance
    print("PERFORMANCE TESTS")
    print("-" * 70)
    test_parallel_speedup()
    test_cache_speedup()
    print()
    
    # Integration
    print("INTEGRATION TESTS")
    print("-" * 70)
    test_static_facts_coverage()
    test_research_result_structure()
    print()
    
    # Stress tests
    print("STRESS TESTS")
    print("-" * 70)
    test_concurrent_queries()
    test_rapid_fire_queries()
    print()
    
    # Print summary
    success = report.print_summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
