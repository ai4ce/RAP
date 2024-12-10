window.HELP_IMPROVE_VIDEOJS = false;

window.addEventListener('load', () => {
    const swiper1 = new Swiper('.swiper1', {
        loop: true,
        autoHeight: true,
        navigation: {
            nextEl: '.swiper1 .swiper-button-next',
            prevEl: '.swiper1 .swiper-button-prev',
        },
        pagination: {
            el: '.swiper1 .swiper-pagination',
            clickable: true,
        },
    });
    const swiper2 = new Swiper('.swiper2', {
        loop: true,
        autoHeight: true,
        navigation: {
            nextEl: '.swiper2 .swiper-button-next',
            prevEl: '.swiper2 .swiper-button-prev',
        },
        pagination: {
            el: '.swiper2 .swiper-pagination',
            clickable: true,
        },
    });
    const swiper3 = new Swiper('.swiper3', {
        loop: true,
        autoHeight: true,
        pagination: {
            el: '.swiper-pagination',
            clickable: true,
        },
        navigation: {
            nextEl: '.swiper-button-next',
            prevEl: '.swiper-button-prev',
        },
        // on: {
        //     init: () => {
        //         centerImageGrid();
        //     },
        //     slideChange: () => {
        //         centerImageGrid();
        //     },
        // },
    });
});

function centerImageGrid() {
    const imageGrid = document.querySelector('.image-grid');
    const swiper = document.querySelector('.swiper3');

    if (imageGrid && swiper) {
        const swiperHeight = swiper.clientHeight;
        const gridHeight = imageGrid.clientHeight;
        const marginTop = (swiperHeight - gridHeight) / 2;

        imageGrid.style.marginTop = marginTop > 0 ? `${marginTop}px` : '0px';
    }
}