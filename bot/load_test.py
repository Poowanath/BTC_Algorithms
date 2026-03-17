"""
Load testing script for BTC Trading Bot
Tests concurrent users and response times
"""
import asyncio
import time
import statistics
from typing import List, Dict
import httpx

# Configuration
BASE_URL = "http://localhost:8000"  # Change to your Render URL for production test
ENDPOINTS = [
    "/health",
    "/predict",
    "/signal?strategy=trend",
    "/signal?strategy=mean_reversion",
    "/signal?strategy=grid",
]


async def make_request(client: httpx.AsyncClient, endpoint: str) -> Dict:
    """Make a single request and measure response time."""
    start_time = time.time()
    try:
        response = await client.get(f"{BASE_URL}{endpoint}", timeout=30.0)
        elapsed = time.time() - start_time
        return {
            "endpoint": endpoint,
            "status": response.status_code,
            "time": elapsed,
            "success": response.status_code == 200,
            "error": None
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "endpoint": endpoint,
            "status": 0,
            "time": elapsed,
            "success": False,
            "error": str(e)
        }


async def simulate_user(user_id: int, num_requests: int = 5) -> List[Dict]:
    """Simulate a single user making multiple requests."""
    results = []
    async with httpx.AsyncClient() as client:
        for _ in range(num_requests):
            # Random endpoint
            import random
            endpoint = random.choice(ENDPOINTS)
            result = await make_request(client, endpoint)
            result["user_id"] = user_id
            results.append(result)
            
            # Wait a bit between requests (simulate real user)
            await asyncio.sleep(random.uniform(0.5, 2.0))
    
    return results


async def load_test(num_users: int, requests_per_user: int = 5):
    """Run load test with specified number of concurrent users."""
    print(f"\n{'='*60}")
    print(f"Load Test: {num_users} concurrent users")
    print(f"Requests per user: {requests_per_user}")
    print(f"Total requests: {num_users * requests_per_user}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Create tasks for all users
    tasks = [simulate_user(i, requests_per_user) for i in range(num_users)]
    
    # Run all users concurrently
    all_results = await asyncio.gather(*tasks)
    
    # Flatten results
    results = [r for user_results in all_results for r in user_results]
    
    total_time = time.time() - start_time
    
    # Analyze results
    analyze_results(results, total_time)


def analyze_results(results: List[Dict], total_time: float):
    """Analyze and print test results."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    response_times = [r["time"] for r in successful]
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total requests: {len(results)}")
    print(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    
    if response_times:
        print(f"\nResponse Times:")
        print(f"  Min: {min(response_times):.3f}s")
        print(f"  Max: {max(response_times):.3f}s")
        print(f"  Mean: {statistics.mean(response_times):.3f}s")
        print(f"  Median: {statistics.median(response_times):.3f}s")
        
        if len(response_times) > 1:
            print(f"  Std Dev: {statistics.stdev(response_times):.3f}s")
    
    print(f"\nRequests per second: {len(results)/total_time:.2f}")
    
    # Endpoint breakdown
    print(f"\nEndpoint Breakdown:")
    endpoint_stats = {}
    for r in results:
        ep = r["endpoint"]
        if ep not in endpoint_stats:
            endpoint_stats[ep] = {"success": 0, "failed": 0, "times": []}
        
        if r["success"]:
            endpoint_stats[ep]["success"] += 1
            endpoint_stats[ep]["times"].append(r["time"])
        else:
            endpoint_stats[ep]["failed"] += 1
    
    for ep, stats in endpoint_stats.items():
        total = stats["success"] + stats["failed"]
        avg_time = statistics.mean(stats["times"]) if stats["times"] else 0
        print(f"  {ep}")
        print(f"    Success: {stats['success']}/{total} ({stats['success']/total*100:.1f}%)")
        print(f"    Avg time: {avg_time:.3f}s")
    
    # Show errors if any
    if failed:
        print(f"\nErrors:")
        error_counts = {}
        for r in failed:
            error = r["error"] or "Unknown"
            error_counts[error] = error_counts.get(error, 0) + 1
        
        for error, count in error_counts.items():
            print(f"  {error}: {count} times")
    
    print(f"{'='*60}\n")


async def main():
    """Run multiple load tests with increasing users."""
    print("BTC Trading Bot - Load Testing")
    print("Make sure the server is running!")
    
    # Test scenarios
    scenarios = [
        (1, 5),    # 1 user, 5 requests
        (5, 5),    # 5 users, 5 requests each
        (10, 5),   # 10 users, 5 requests each
        (20, 3),   # 20 users, 3 requests each
        (50, 2),   # 50 users, 2 requests each (stress test)
    ]
    
    for num_users, requests_per_user in scenarios:
        await load_test(num_users, requests_per_user)
        
        # Wait between tests
        if num_users < 50:
            print("Waiting 5 seconds before next test...\n")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
