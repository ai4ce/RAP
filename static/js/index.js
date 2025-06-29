document.addEventListener('DOMContentLoaded', () => {
  // 初始化所有轮播
  const splide1 = new Splide('#video-carousel1', {
    type: 'loop',
    perPage: 1,
    autoplay: false
  });
  
  const splide2 = new Splide('#video-carousel2', {
    type: 'loop',
    perPage: 1,
    autoplay: false
  });
  
  const splide3 = new Splide('#video-carousel3', {
    type: 'loop',
    perPage: 1,
    autoplay: false
  });
  
  splide1.mount();
  splide2.mount();
  splide3.mount();


  splide3.on('move', function(newIndex) {
    if (newIndex === 5) {
        const imgs = document.querySelectorAll('.image-grid img')
        imgs.forEach((img, index) => {
            img.style.display = 'block';
        });
    }
  });

  splide3.on('moved', function(newIndex) {
    if (newIndex !== 5) {
        const imgs = document.querySelectorAll('.image-grid img')
        imgs.forEach((img, index) => {
            img.style.display = 'none';
        });
    }
  });
});