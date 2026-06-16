// JS glue satisfying platform.zig's swr_js_* wasm imports via --js-library.
// __proxy:'sync' runs them on the browser main thread (renderer is on a worker
// pthread) and blocks until the DOM work completes.
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
    // HEAPU8 is a SharedArrayBuffer view: main thread reads the worker's framebuffer.
    cache.image.data.set(HEAPU8.subarray(ptr, ptr + w * h * 4));
    cache.ctx.putImageData(cache.image, 0, 0);
  },
});
