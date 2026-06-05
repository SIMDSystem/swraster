// web_zig_lib.js — JS glue for the Zig wasm build.
//
// The C++ web backend embeds this logic inline via MAIN_THREAD_EM_ASM. The Zig
// backend instead declares swr_js_setup_canvas / swr_js_present as imports
// (platform.zig) and we satisfy them here through emcc's --js-library. Both are
// marked __proxy: 'sync' so they run on the browser main thread (the renderer
// lives on a worker pthread under PROXY_TO_PTHREAD) and block the caller until
// the DOM work completes — exactly matching the C++ MAIN_THREAD_EM_ASM blit.
addToLibrary({
  swr_js_setup_canvas__proxy: 'sync',
  swr_js_setup_canvas__sig: 'vii',
  swr_js_setup_canvas: function (w, h) {
    var canvas = Module.canvas;
    if (!canvas) {
      canvas = document.getElementById('canvas');
      if (canvas) Module.canvas = canvas;
    }
    if (canvas) {
      canvas.width = w;
      canvas.height = h;
      canvas.addEventListener('wheel', function (ev) { ev.preventDefault(); }, { passive: false });
      canvas.addEventListener('contextmenu', function (ev) { ev.preventDefault(); });
      if (!canvas.hasAttribute('tabindex')) canvas.setAttribute('tabindex', '0');
    }
  },

  swr_js_present__proxy: 'sync',
  swr_js_present__sig: 'viii',
  swr_js_present: function (ptr, w, h) {
    var canvas = Module.canvas;
    if (!canvas) {
      canvas = document.getElementById('canvas');
      if (canvas) Module.canvas = canvas;
    }
    if (!canvas) return;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    var cache = Module.swrCanvasCache;
    if (!cache) {
      cache = {};
      Module.swrCanvasCache = cache;
    }
    if (cache.canvas !== canvas || cache.w !== w || cache.h !== h) {
      cache.canvas = canvas;
      cache.w = w;
      cache.h = h;
      cache.ctx = canvas.getContext('2d');
      cache.image = cache.ctx.createImageData(w, h);
    }
    // HEAPU8 is a SharedArrayBuffer view, so the main thread can read the
    // framebuffer the worker just wrote.
    cache.image.data.set(HEAPU8.subarray(ptr, ptr + w * h * 4));
    cache.ctx.putImageData(cache.image, 0, 0);
  },
});
