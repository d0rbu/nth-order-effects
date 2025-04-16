const queryArg = "section";

function syncHFSpacesURLHash() {
  // Handle explicit section requests (don't update hash automatically on load)
  const hasExplicitRequest = handleExplicitSectionRequest();
  
  // Set up hash change monitoring
  updateHashBasedOnHashChange();
  
  // Always set up scroll monitoring to update hash during scrolling
  setupScrollMonitoring();
  
  // If no explicit request, we don't update the hash on initial load
  // The hash will only start updating when the user scrolls
}

function handleExplicitSectionRequest() {
  // Check for section parameter in URL
  const urlParams = new URLSearchParams(window.location.search);
  const sectionId = urlParams.get(queryArg);
  
  // If we have an explicit section request
  if (sectionId) {
    const targetElement = document.getElementById(sectionId);
    if (targetElement) {
      // Slight delay to ensure the browser doesn't try to do its own scrolling first
      setTimeout(() => {
        targetElement.scrollIntoView();
        history.replaceState(null, null, `#${sectionId}`);
      }, 100);
    }
    return true;
  }
  
  // No explicit section parameter found
  return false;
}

function setupScrollMonitoring() {
  // Variables to manage throttling
  let isScrolling = false;
  let lastKnownScrollPosition = 0;
  let initialScroll = true;
  
  // Add the scroll event listener
  window.addEventListener('scroll', function() {
    lastKnownScrollPosition = window.scrollY;
    
    if (!isScrolling) {
      window.requestAnimationFrame(function() {
        // Skip the first scroll event which might be browser's automatic scroll
        // to a hash on page load
        if (initialScroll) {
          initialScroll = false;
        } else {
          updateHashBasedOnScroll(lastKnownScrollPosition);
        }
        isScrolling = false;
      });
    }
    
    isScrolling = true;
  });
}

// Function to update the URL hash based on scroll position
function updateHashBasedOnScroll(scrollPosition) {
  const closestHeading = findClosestHeading(scrollPosition);
  
  // Update the URL hash if we found a closest element
  if (closestHeading && closestHeading.id) {
    // Only update if the hash is different to avoid unnecessary operations
    if (window.location.hash !== `#${closestHeading.id}`) {
      silentlyUpdateHash(closestHeading.id);
      postMessageToHFSpaces(closestHeading.id);
    }
  }
}

// Find the closest heading to the current scroll position
function findClosestHeading(scrollPosition) {
  // Get only heading elements with IDs that we want to track
  const headingsWithIds = Array.from(document.querySelectorAll('h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]'));

  // Skip if there are no headings with IDs
  if (headingsWithIds.length === 0) return null;

  // Find the element closest to the middle of the viewport
  let closestHeading = null;
  let closestDistance = Infinity;
  const viewportMiddle = scrollPosition + window.innerHeight / 2;
  
  // Iterate through all headings to find the closest one
  headingsWithIds.forEach(heading => {
    const headingTop = heading.getBoundingClientRect().top + scrollPosition;
    const distance = Math.abs(headingTop - viewportMiddle);
    
    if (distance < closestDistance) {
      closestDistance = distance;
      closestHeading = heading;
    }
  });
  
  return closestHeading;
}

// Update hash without triggering scroll or other side effects
function silentlyUpdateHash(id) {
  history.replaceState(null, null, `#${id}`);
}

function updateHashBasedOnHashChange() {
  window.addEventListener('hashchange', () => {
    const elementId = window.location.hash.slice(1);
    postMessageToHFSpaces(elementId);
  });
}

function postMessageToHFSpaces(elementId) {
  const parentOrigin = "https://huggingface.co";
  window.parent.postMessage({ queryString: `${queryArg}=${elementId}` }, parentOrigin);
}

export { syncHFSpacesURLHash };
