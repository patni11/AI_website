
"use strict";
const htmlTag = document.querySelector('html');
const nav_bar = document.querySelector('nav');
const bodyTag = document.querySelector('body');

let scrolled = () => {
   let h = scrollY / (bodyTag.scrollHeight - innerHeight);
   return Math.floor(h * 100);
}

addEventListener('scroll', () => {
   nav_bar.style.setProperty('background', scrolled > 250? "var(--color2)" : "var(--color1)");
})