async function loadFragments() {
    // Find all elements with ids starting with 'fragment-'
    const fragmentElements = Array.from(document.querySelectorAll('[id^="fragment-"]'));
    
    class FetchQueue {
        constructor(maxConcurrent = 3) {
            this.queue = [];
            this.maxConcurrent = maxConcurrent;
            this.activeFetches = 0;
            this.maxRetries = 3; // Maximum number of retry attempts
            this.baseDelay = 1000; // Base delay in milliseconds (1 second)
        }

        async sleep(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }

        async fetchWithRetry(fragmentPath, retryCount = 0) {
            try {
                const response = await fetch(fragmentPath);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return await response.text();
            } catch (error) {
                if (retryCount < this.maxRetries) {
                    // Exponential backoff: 1s, 2s, 4s
                    const delay = this.baseDelay * Math.pow(2, retryCount);
                    console.warn(`Retry ${retryCount + 1}/${this.maxRetries} for ${fragmentPath} after ${delay}ms`);
                    await this.sleep(delay);
                    return this.fetchWithRetry(fragmentPath, retryCount + 1);
                }
                throw error;
            }
        }

        async addFetch(element) {
            const fragmentName = element.id.replace('fragment-', '');
            const fragmentPath = `fragments/${fragmentName}.html`;
            
            return new Promise(async (resolve, reject) => {
                try {
                    const fetchPromise = (async () => {
                        try {
                            const html = await this.fetchWithRetry(fragmentPath);
                            
                            // Process the fragment
                            const temp = document.createElement('div');
                            temp.innerHTML = html;
                            element.innerHTML = temp.innerHTML;
                            
                            // Handle scripts
                            const scripts = temp.getElementsByTagName('script');
                            Array.from(scripts).forEach(oldScript => {
                                const newScript = document.createElement('script');
                                Array.from(oldScript.attributes).forEach(attr => {
                                    newScript.setAttribute(attr.name, attr.value);
                                });
                                newScript.textContent = oldScript.textContent;
                                oldScript.parentNode.removeChild(oldScript);
                                document.body.appendChild(newScript);
                            });
                            
                            this.activeFetches--;
                            resolve();
                        } catch (error) {
                            console.error(`Failed to load fragment ${fragmentPath} after ${this.maxRetries} retries:`, error);
                            this.activeFetches--;
                            reject(error);
                        }
                    })();

                    this.queue.push(fetchPromise);
                    this.activeFetches++;
                } catch (error) {
                    reject(error);
                }
            });
        }

        async processNext(element) {
            if (this.activeFetches < this.maxConcurrent && element) {
                await this.addFetch(element);
            }
        }
    }

    // Initialize queue
    const fetchQueue = new FetchQueue(3);
    let currentIndex = 0;
    const elements = fragmentElements; // Assuming this is defined elsewhere

    // Initial loading of first 3 elements
    while (currentIndex < elements.length && currentIndex < 3) {
        await fetchQueue.processNext(elements[currentIndex]);
        currentIndex++;
    }

    // Process remaining elements as fetches complete
    while (currentIndex < elements.length) {
        // Wait for any fetch to complete
        await Promise.race(fetchQueue.queue);
        // Remove completed fetch from queue
        fetchQueue.queue = fetchQueue.queue.filter(p => p.status === 'pending');
        // Add next element to queue
        await fetchQueue.processNext(elements[currentIndex]);
        currentIndex++;
    }

    // Wait for remaining fetches to complete
    await Promise.all(fetchQueue.queue);
}

export { loadFragments }