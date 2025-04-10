# Continuous Fermat Spiral (CFS) Fill Algorithm Plan


## Algorithm Steps:

1.  **Input:** A singly connected 2D region `R` representing a single slice layer (e.g., a `shapely.Polygon`).
2.  **Parameters:** Define the desired toolpath width `w` (spacing between adjacent path segments).
3.  **Generate Iso-contours:**
    *   Compute successive inward offsets (negative buffers) of the region boundary `∂R` at distances `(i - 0.5)w` for `i = 1, 2, ...` until the offset is empty or invalid. This generates the set of iso-contours `L`.
    *   Use a library like Shapely (`polygon.buffer(-offset)`) for offsetting. Handle potential `MultiPolygon` results at each offset level.
    *   Index the resulting contours `c_i,j`, where `i` is the offset level (distance index) and `j` distinguishes multiple disjoint contours at the same level. `c_1,1` is the original boundary `∂R`.
4.  **Build Initial Connectivity Graph:**
    *   Create a graph where nodes represent the iso-contours `c_i,j`.
    *   Add edges between contours at adjacent levels (`c_i,j` and `c_i+1,j'`).
    *   Determine the "connecting segment" `O_i,j,j'` on `c_i,j` consisting of points closer to `c_i+1,j'` than any other contour `c_i+1,k` at level `i+1`. This might involve proximity analysis (e.g., using Voronoi diagrams or nearest point calculations between contours).
    *   Add an edge only if the connecting segment `O_i,j,j'` is non-empty.
    *   Assign the geometric length of `O_i,j,j'` as the weight for the edge (`c_i,j`, `c_i+1,j'`). Lower weight indicates a preferred connection point (shorter rerouting segment).
5.  **Construct Spiral-Contour Tree:**
    *   Compute the Minimum Spanning Tree (MST) of the connectivity graph using an algorithm like Prim's or Kruskal's. Start the MST from the root node `c_1,1` (the original boundary).
6.  **Identify Spirallable Regions & Branch Points:**
    *   Traverse the MST.
    *   Nodes with degree <= 2 (in the MST) are part of paths corresponding to "spirallable regions".
    *   Nodes with degree > 2 are branch points where multiple regions/spirals merge.
7.  **Recursive Rerouting (Bottom-Up Traversal of MST):**
    *   Process the tree from the leaves towards the root `c_1,1`.
    *   **Leaf Nodes:** Represent the innermost contours.
    *   **Spirallable Segments (Degree 1/2 Nodes):** For sequences of nodes forming a path in the MST, apply the Fermat spiral generation logic (Section 3 of the paper). This involves defining inward/outward links and rerouting points to convert the sequence of offset contours into a single inward-then-outward spiral segment connecting the entry/exit points defined by the MST connections.
    *   **Branch Nodes (Degree > 2 Nodes):** Implement logic to merge the incoming continuous paths from the child branches in the MST. The connection points are determined by the edges (and associated connecting segments `O`) of the MST.
    *   **Root Node:** The final result is a single continuous path starting and ending on the boundary `c_1,1`.
8.  **Output:** The final generated continuous Fermat spiral toolpath for the layer `R`.
9.  **(Post-processing):** Apply smoothing or optimization to remove potential "jaggies" introduced during the rerouting process (as mentioned briefly in the paper).

## Implementation Notes:

*   Requires robust geometric operations (offsetting, intersection, distance, length) - Shapely is a good candidate.
*   Graph representation and MST algorithm (e.g., using `networkx`).
*   Detailed implementation of the Fermat spiral rerouting logic (inward/outward links, point selection `B(p)`, `N(p)`) needs careful translation from the paper's description and figures.
*   Handling numerical precision issues in geometric computations is crucial.
