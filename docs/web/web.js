const revealElements = Array.from(document.querySelectorAll(".reveal"));

function revealOnScroll() {
  const threshold = window.innerHeight * 0.9;
  for (const element of revealElements) {
    const top = element.getBoundingClientRect().top;
    if (top < threshold) {
      element.classList.add("revealed");
    }
  }
}

window.addEventListener("scroll", revealOnScroll, { passive: true });
window.addEventListener("load", revealOnScroll);
