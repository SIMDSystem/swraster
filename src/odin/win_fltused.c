// Odin's windows_amd64 objects reference the MSVC `_fltused` float-usage marker,
// which zig's bundled mingw CRT does not define. Provide it for the COFF link.
int _fltused = 0;
