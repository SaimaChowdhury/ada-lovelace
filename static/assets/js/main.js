(function ($) {
  "use strict";


  $( function() {
    var handle = $( "#custom-handle" );
    $( "#slider" ).slider({
      create: function() {
        handle.text( $( this ).slider( "value" ) );
      },
      slide: function( event, ui ) {
        handle.text( ui.value );
      }
    });
  } );

   /***********
   Preloader
   ************/

  function preLoader() {
    window.onload = function () {
      var preloader = document.querySelector(".preloader");
      if (preloader !== null) {
        preloader.classList.add("animate__animated", "animate__fadeOut");
        setTimeout(function () {
          preloader.style.display = "none";
        }, 200);
      }
    };
  }
  preLoader();



   /***********
   Custom Cursor
   ************/
  function customCursor() {
    var c = document.querySelector(".cursor-dot");

    if (c !== null) {
      var cursor = {
        delay: 7,
        _x: 0,
        _y: 0,
        endX: window.innerWidth / 2,
        endY: window.innerHeight / 2,
        cursorVisible: true,
        cursorEnlarged: false,
        $dot: document.querySelector(".cursor-dot"),
        $outline: document.querySelector(".cursor-dot-outline"),

        init: function () {
          // Set up element sizes
          this.dotSize = this.$dot.offsetWidth;
          this.outlineSize = this.$outline.offsetWidth;

          this.setupEventListeners();
          this.animateDotOutline();
        },

        updateCursor: function (e) {
          var self = this;

          // Show the cursor
          self.cursorVisible = true;
          self.toggleCursorVisibility();

          // Position the dot
          self.endX = e.clientX;
          self.endY = e.clientY;
          self.$dot.style.top = self.endY + "px";
          self.$dot.style.left = self.endX + "px";
        },

        setupEventListeners: function () {
          var self = this;

          // Reposition cursor on window load
          window.addEventListener("load", (event) => {
            self.cursorEnlarged = false;
            self.toggleCursorSize();
          });

          // Anchor hovering
          document.querySelectorAll("a, button").forEach(function (el) {
            el.addEventListener("mouseover", function () {
              self.cursorEnlarged = true;
              self.toggleCursorSize();
            });
            el.addEventListener("mouseout", function () {
              self.cursorEnlarged = false;
              self.toggleCursorSize();
            });
          });

          // Click events
          document.addEventListener("mousedown", function () {
            self.cursorEnlarged = true;
            self.toggleCursorSize();
          });
          document.addEventListener("mouseup", function () {
            self.cursorEnlarged = false;
            self.toggleCursorSize();
          });

          document.addEventListener("mousemove", function (e) {
            // Show the cursor
            self.cursorVisible = true;
            self.toggleCursorVisibility();

            // Position the dot
            self.endX = e.clientX;
            self.endY = e.clientY;
            self.$dot.style.top = self.endY + "px";
            self.$dot.style.left = self.endX + "px";
          });

          // Hide/show cursor
          document.addEventListener("mouseenter", function (e) {
            self.cursorVisible = true;
            self.toggleCursorVisibility();
            self.$dot.style.opacity = 1;
            self.$outline.style.opacity = 1;
          });

          document.addEventListener("mouseleave", function (e) {
            self.cursorVisible = true;
            self.toggleCursorVisibility();
            self.$dot.style.opacity = 0;
            self.$outline.style.opacity = 0;
          });
        },

        animateDotOutline: function () {
          var self = this;

          self._x += (self.endX - self._x) / self.delay;
          self._y += (self.endY - self._y) / self.delay;
          self.$outline.style.top = self._y + "px";
          self.$outline.style.left = self._x + "px";

          requestAnimationFrame(this.animateDotOutline.bind(self));
        },

        toggleCursorSize: function () {
          var self = this;

          if (self.cursorEnlarged) {
            self.$dot.style.transform = "translate(-50%, -50%) scale(0.75)";
            self.$outline.style.transform = "translate(-50%, -50%) scale(1.6)";
          } else {
            self.$dot.style.transform = "translate(-50%, -50%) scale(1)";
            self.$outline.style.transform = "translate(-50%, -50%) scale(1)";
          }
        },

        toggleCursorVisibility: function () {
          var self = this;

          if (self.cursorVisible) {
            self.$dot.style.opacity = 1;
            self.$outline.style.opacity = 1;
          } else {
            self.$dot.style.opacity = 0;
            self.$outline.style.opacity = 0;
          }
        },
      };

      cursor.init();
    }
  }
  customCursor();


 

  /***********
   mobile menu  js
   ************/
  $(".hamburger").on("click", function (event) {
    $(this).toggleClass("h-active");
    $(".main-nav").toggleClass("slidenav");
  });

  $(".header-home .main-nav ul li  a").on("click", function (event) {
    $(".hamburger").removeClass("h-active");
    $(".main-nav").removeClass("slidenav");
  });

  $(".main-nav .fl").on("click", function (event) {
    var $fl = $(this);
    $(this).parent().siblings().find(".sub-menu").slideUp();
    $(this).parent().siblings().find(".fl").addClass("flaticon-plus").text("+");
    if ($fl.hasClass("flaticon-plus")) {
      $fl.removeClass("flaticon-plus").addClass("flaticon-minus").text("-");
    } else {
      $fl.removeClass("flaticon-minus").addClass("flaticon-plus").text("+");
    }
    $fl.next(".sub-menu").slideToggle();
  });

  /* ========================== 
   Sticky Navbar
   ==========================*/

  $(window).scroll(function () {
    var scroll = $(window).scrollTop();
    if (scroll >= 20) {
      $(".header-area").addClass("sticky");
    } else {
      $(".header-area").removeClass("sticky");
    }
  });


   /***********
   Back To Top
   ************/
   function backToTop() {
    var scrollpos = window.scrollY;
    var backBtn = document.querySelector(".back-top");

    if (backBtn !== null) {
      var addClassOnScroll = function () {
        backBtn.classList.add("back-top-show");
      };

      var removeClassOnScroll = function () {
        backBtn.classList.remove("back-top-show");
      };

      window.addEventListener("scroll", function () {
        scrollpos = window.scrollY;
        if (scrollpos >= 500) {
          addClassOnScroll();
        } else {
          removeClassOnScroll();
        }
      });

      backBtn.addEventListener("click", function () {
        window.scrollTo({
          top: 0,
          behavior: "smooth",
        });
      });
    }
  }
  backToTop();

  /* ========================== 
   WOW JS
   ==========================*/
  new WOW({
    offset: 100,
    animateClass: "animated",
    mobile: true,
  }).init();



  /* ========================== 
   Highlighted Text
   ==========================*/
  var detail = $(".text_gradient");
  var offset = 0;
  detail.each(function () {
    var element = $(this);
    $(window).on("scroll", function () {
      offset = $(window).scrollTop();
      var h = $(window).height();
      var i = element.offset().top - offset - h;
      i *= -1;
      var y = (i * 240) / h;
      element.css({ backgroundSize: y + "% 100%" });
    });
  });

  /* ========================== 
   Counter Up
   ==========================*/
  function tm_mycounter() {
    jQuery(".tm_counter").removeClass("stop");
    jQuery(".tm_counter").each(function () {
      var el = jQuery(this);
      el.waypoint({
        handler: function () {
          if (!el.hasClass("stop")) {
            el.addClass("stop").countTo({
              refreshInterval: 50,
              formatter: function (value, options) {
                return value
                  .toFixed(options.decimals)
                  .replace(/\B(?=(?:\d{3})+(?!\d))/g, ",");
              },
            });
          }
        },
        offset: "85%",
      });
    });
  }
  tm_mycounter();


  /* ========================== 
   Navarbar Active Link
   ==========================*/
  const sections = document.querySelectorAll("section[id]");
  const navigationLinks = document.querySelectorAll(".navigation a");
  
  window.addEventListener("scroll", navHighlighter);
  
  function navHighlighter() {
    const scrollY = window.pageYOffset;
  
    sections.forEach(current => {
      const sectionId = current.getAttribute("id");
      const sectionTop = current.offsetTop - window.innerHeight / 1.5;
      const sectionHeight = current.offsetHeight;
  
      if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
        navigationLinks.forEach(link => {
          const linkId = link.getAttribute("href").substring(1); // Remove the #
          if (sectionId === linkId) {
            link.classList.add("section-active");
          } else {
            link.classList.remove("section-active");
          }
        });
      }
    });
  }
  
 
  /*========================
    Swiper Slider
    =========================*/
  var testimonialSlider = new Swiper(".testimonial-slider", {
    spaceBetween: 30,
    loop: true,
    autoplay: {
      delay: 2500,
      disableOnInteraction: false,
    },
    pagination: {
      el: ".testimonial-pagination",
      clickable: true,
    },
    breakpoints: {
      320: {
        slidesPerView: 1.2,
      },
      567: {
        slidesPerView: 2,
      },
      992: {
        slidesPerView: 3,
      },
    },
  });

  var portfoioSlider = new Swiper(".portfolio-slider", {
    spaceBetween: 30,
    autoplay: {
      delay: 2500,
      disableOnInteraction: false,
    },
    loop: true,
    pagination: {
      el: ".portfolio-pagination",
      clickable: true,
    },
    breakpoints: {
      320: {
        slidesPerView: 1.2,
      },
      567: {
        slidesPerView: 2.2,
      },
      776: {
        slidesPerView: 2.8,
      },
      992: {
        slidesPerView: 3.5,
      },
      2200: {
        slidesPerView: 4.5,
      },
    },
  });
})(jQuery);
