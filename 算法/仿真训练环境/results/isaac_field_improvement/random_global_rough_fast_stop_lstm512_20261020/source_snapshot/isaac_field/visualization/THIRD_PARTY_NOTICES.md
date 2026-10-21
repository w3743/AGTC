# Third-party notices

## llm-viz

This viewer adapts small portions of the `llm-viz` project by Brendan Bycroft,
available at <https://github.com/bbycroft/llm-viz>, pinned locally at commit
`9da93742382f1bf36c020c38a1ace454e82c4490`.

Adapted source files and their use in this viewer:

- `src/utils/bezier.ts`: the cubic Bézier Bernstein-polynomial evaluation in
  `bezierCubic` is adapted to two dimensions by
  `isaac_field/visualization/network_layout.py::_cubic_sample` for route
  diagnostics. The emitted SVG routes are cubic Béziers as in the source
  viewer's vector drawing approach.
- `src/llm/Sidebar.tsx` and `src/llm/Sidebar.module.scss`: earlier versions
  adapted grouped region navigation and its compact sidebar. The current
  page has removed that navigation and its dedicated styles and handlers;
  the pinned sources remain for historical provenance.
- `src/llm/components/TocDiagram.tsx` and
  `src/llm/components/TocDiagram.module.scss`: the node active/hover
  selection treatment is adapted in `network_2d.html`. Earlier path
  highlighting and dimming have been removed; real edges keep their
  fixed branch colors and all remain present.

The viewer does not import the upstream project at runtime and does not copy
its homepage, third-party font assets, 3D network rendering, or model
computation. Layout, content, colors, and interaction details are adapted for
this project's live 2D policy-network data.

The upstream project's MIT license follows:

```text
MIT License

Copyright (c) 2023-2026 Brendan Bycroft

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
