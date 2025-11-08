# """Asynchronous request module"""
# from concurrent import futures
# import itertools
# import requests
# import time

# class AsyncRequester(object):
#     """Send asynchronous requests to list of urls
#     Response time may be limited by rate of system/NW"""
#     def __init__(self, n_threads):
#         # self.urllist = []
#         self.n_threads = n_threads
#         self.executor = futures.ThreadPoolExecutor(max_workers=self.n_threads)

#     def send_async_requests(self, urllist):
#         """Sends asynchronous requests to a list of urls """
#         if len(urllist) == 1:
#             return self.get_batch(urllist)
#         # # Num of batches per thread
#         # n_batches = int(len(urllist) / self.n_threads) + 1
#         # # List of URLs per batch
#         # batch_urllist = [urllist[i * n_batches: (i + 1) * n_batches]
#         #                 for i in range(self.n_threads)]
#         # # List of HTTP response
#         # responses = list(self.executor.map(self.get_batch, batch_urllist))
#         # return list(itertools.chain(*responses))    # Return 1 sequence of responses

#         batch_size = 100  # Very small batches
#         batch_urllist = [urllist[i:i+batch_size] for i in range(0, len(urllist), batch_size)]
        
#         responses = []
#         for i, batch in enumerate(batch_urllist):
#             if i % 10 == 0:  # Progress update every 10 batches
#                 print(f"Processing batch {i+1}/{len(batch_urllist)} ({len(responses)}/{len(urllist)} requests completed)")
            
#             batch_responses = list(self.executor.map(self.get_json, batch))
#             responses.extend(batch_responses)
            
#             # Pause between batches to avoid overwhelming the system
#             time.sleep(0.5)
        
#         return responses

#     def get_json(cls, url):
#         """open URL and return JSON contents"""
#         result = requests.get(url).json()
#         return result

#     def get_batch(self, urllist):
#         """Batch processing for get method; takes list of urls as input"""
#         return [self.get_json(url) for url in urllist]
"""Asynchronous request module"""
from concurrent import futures
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class AsyncRequester(object):
    """Send asynchronous requests to list of urls
    Response time may be limited by rate of system/NW"""
    
    def __init__(self, n_threads):
        self.n_threads = n_threads
        self.executor = futures.ThreadPoolExecutor(max_workers=self.n_threads)
        
        # Create a session with connection pooling
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        # Configure adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.n_threads * 2,
            pool_maxsize=self.n_threads * 4,
            pool_block=False
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def send_async_requests(self, urllist):
        """Sends asynchronous requests to a list of urls"""
        if len(urllist) == 1:
            return [self.get_json(urllist[0])]
        
        # Use the thread pool more efficiently - submit all at once
        # instead of batching manually
        responses = list(self.executor.map(self.get_json, urllist))
        return responses
    
    def get_json(self, url):
        """Open URL and return JSON contents with connection reuse"""
        try:
            result = self.session.get(url, timeout=10).json()
            return result
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            # Retry once
            time.sleep(0.5)
            try:
                result = self.session.get(url, timeout=10).json()
                return result
            except Exception as e2:
                print(f"Retry failed for {url}: {e2}")
                raise
    
    def get_batch(self, urllist):
        """Batch processing for get method; takes list of urls as input"""
        return [self.get_json(url) for url in urllist]