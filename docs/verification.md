# Verification record

Completed on 21 October 2026 using Python 3.12.14, NumPy 2.5.4 and OpenCV headless 5.0.0.93. Direct dependencies are pinned in `requirements.txt`.

## Source checks

The local Python CV script and its RMI submission copy have the same SHA-256 hash: `87d917ade1208bb5d7e994f213e5630cb7f18bba2a58db526a0d7f170a57526e`. One unchanged copy is archived. The example JPEG also matches its original bytes. Hashes are recorded in `source-hashes.json`.

## End-to-end result

Ran the current CLI against the supplied JPEG. It decoded the image, detected a 6×6 grid, identified start `(5,5)` and goal `(0,0)`, found a route, and saved the mask, annotated image and JSON.

The route contains 11 cells / 10 moves and has cost 23 under the documented entered-cell convention. Every move is orthogonal, every visited cell is traversable, and the cost recomputed from the visited cells is 23. Dijkstra's algorithm returned the same minimum cost. The output image was visually inspected and the path follows the identified cells.

The CLI was also launched from outside the repository using its bundled default input and a separate output folder. The resulting `route.json` had the same SHA-256 hash as the repository example result.

## Automated checks

Ran `python -m unittest discover -s tests -v` with temporary outputs directed into a writable workspace. All eight test methods passed:

1. Supplied-image extraction, endpoint detection, search cost, path validity and saved outputs.
2. A weighted detour that costs less than the geometrically shorter route.
3. 500 random 2–7 row/column grids, seeded with 2841, compared against a separate cost-only Dijkstra search. Cases include blocked cells, zero costs, fractional costs and unreachable targets.
4. Unreachable grids and blocked endpoints.
5. A coincident start/goal and zero-cost interior cells.
6. Empty/ragged grids, negative/nonfinite weights and invalid endpoints.
7. Missing and duplicate start markers in edited example images.
8. Undetectable grid boundaries and invalid cell-sampling margins.

For reachable random cases, path checks verify endpoints, no repeated cells, legal four-neighbor steps, traversability and exact entered-cell cost. Unreachable results agree with Dijkstra.

## Scope

This verifies the current prepared implementation and supplied example, with broader algorithm checks on synthetic cost grids. It does not establish robust grid/color extraction under lighting changes, compression changes, perspective or arbitrary map imagery. Classification errors remain possible because nearest-reference color matching has no rejection threshold.

The original archived script was not executed during preparation. The new results were produced by `planner.py`; they are not presented as historical submission output. No camera or robot was involved.
