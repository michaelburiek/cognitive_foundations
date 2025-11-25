import os
import matplotlib.pyplot as plt
import networkx as nx
import colorsys
import pygraphviz as pgv
from collections import Counter
from typing import List, Tuple


# ----------------------------------------------------------------------------------------------------------------------
# Methods for identifying the most common/consensus or success-prone tree structure
# from multiple successful reasoning traces.

class ConsensusTreeFinder:
    """Find consensus element tree structures from multiple successful traces."""
    
    def __init__(self, span_tree):
        """
        Initialize with a SpanTree instance.
        
        Args:
            span_tree: SpanTree instance with loaded data
        """
        self.span_tree = span_tree

    def construct_consensus_tree(self, problem_type: str, max_nodes: int = 10, success_only: bool = True, dynamic: bool = False) -> nx.DiGraph:
        """
        Construct a consensus tree of elements for a given problem type based on NPMI.
        
        The algorithm:
        1. Find the element with highest NPMI that appears first (earliest start) in successful traces
        2. From current element A, find the edge (A->B) with highest NPMI (any type: contains/next/parallel)
        3. Add edge to graph and move to element B
        4. Repeat until max_nodes reached or no more valid edges
        
        Args:
            problem_type: The problem type to analyze
            max_nodes: Maximum number of nodes to include in the consensus tree
            
        Returns:
            NetworkX DiGraph representing the consensus reasoning path
        """
        # Compute NPMI for elements and edges
        element_npmi = self.span_tree.compute_npmi_node(problem_type=problem_type, 
                                        all_elements=list(self.span_tree.nodes.keys()))
        edge_npmi = self.span_tree.compute_npmi_edge(problem_type=problem_type)
        
        if not element_npmi or not edge_npmi:
            print("No NPMI data available")
            return nx.DiGraph()
        
        # Get successful traces for this problem type
        traces = self._get_all_traces_type(problem_type=problem_type, success_only=success_only)
        
        if not traces:
            print("No traces found")
            return nx.DiGraph()
        
        # Find elements that appear first (earliest start) in successful traces
        first_elements = Counter()
        
        for trace_key in traces:
            if trace_key not in self.span_tree.trace_elements:
                continue
            
            elements = self.span_tree.trace_elements[trace_key]
            if not elements:
                continue
            
            # Sort by start position and get the first element
            sorted_elements = sorted(elements, key=lambda x: (x[1], -(x[2]-x[1])))
            first_element = sorted_elements[0][0]
            first_elements[first_element] += 1
        
        if not first_elements:
            print("No first elements found")
            return nx.DiGraph()
        
        if success_only:
            # Pick the first element with highest NPMI
            candidate_first = [(b, element_npmi[b]['npmi']) 
                            for b in first_elements.keys() 
                            if b in element_npmi]
        else:
            # Pick the first element with highest count
            candidate_first = [(b, first_elements[b]) 
                            for b in first_elements.keys() 
                            if b in element_npmi]
        
        if not candidate_first:
            print("No valid first elements with NPMI scores")
            return nx.DiGraph()
        
        # Start with the element that has highest NPMI (or overall frequency if all traces) among first elements
        current_element = max(candidate_first, key=lambda x: x[1])[0]
        
        # Build consensus graph
        consensus_graph = nx.DiGraph()
        consensus_graph.add_node(current_element, 
                                npmi=element_npmi[current_element]['npmi'],
                                is_start=True)
        
        visited = {current_element}
        
        print(f"Starting consensus tree with: {current_element} (NPMI: {element_npmi[current_element]['npmi']:.3f}, Node Prob: {element_npmi[current_element]['p_element']:.3f})")
        
        # Iteratively add nodes based on highest NPMI edges
        while (len(consensus_graph.nodes()) < max_nodes):
            # Find all outgoing edges from current element
            candidate_edges = []
            
            for (node_a, node_b, edge_type), npmi_data in edge_npmi.items():
                # Must originate from current element
                if node_a != current_element:
                    continue
                
                # Skip self-loops
                if node_a == node_b:
                    continue
                
                # Skip if target already visited (avoid cycles)
                if node_b in visited:
                    continue
                
                candidate_edges.append((node_b, edge_type, npmi_data['npmi'], npmi_data['p_edge']))
            
            if not candidate_edges:
                print(f"No more valid edges from {current_element}")
                break
            
            # Pick edge with highest NPMI
            if success_only:
                next_element, edge_type, edge_npmi_score, edge_prob_score = max(candidate_edges, key=lambda x: x[2])
            else:
                next_element, edge_type, edge_npmi_score, edge_prob_score = max(candidate_edges, key=lambda x: x[3])

            if success_only and dynamic and (edge_npmi_score <= 0):
                break
            
            # Add to graph
            consensus_graph.add_node(next_element, 
                                    npmi=element_npmi[next_element]['npmi'],
                                    p_element=element_npmi[next_element]['p_element'])
            consensus_graph.add_edge(current_element, next_element, 
                                    edge_type=edge_type,
                                    npmi=edge_npmi_score,
                                    p_edge=edge_prob_score)
            
            print(f"  Added edge: {current_element} --[{edge_type}]--> {next_element} "
                  f"(Edge NPMI: {edge_npmi_score:.3f}, Edge Prob: {edge_prob_score:.3f}, Node NPMI: {element_npmi[next_element]['npmi']:.3f}, Node Prob: {element_npmi[next_element]['p_element']:.3f}")
            
            visited.add(next_element)
            current_element = next_element
        
        print(f"\nConsensus tree complete: {len(consensus_graph.nodes())} nodes, {len(consensus_graph.edges())} edges")
        
        return consensus_graph

    def _get_all_traces_type(self, problem_type: str, success_only: bool = True) -> List[Tuple[str, str]]:
        """Get all successful (model, problem) traces for a problem type."""

        all_traces = []

        for problem_id, problem_info in self.span_tree.problems.items():
            if problem_info.problem_type != problem_type:
                continue
            
            if success_only:
                models = problem_info.get_correct_models()
            else:
                models = list(problem_info.correctness.keys())
            
            for model_name in models:
                if (model_name, problem_id) in self.span_tree.trace_elements:
                    all_traces.append((model_name, problem_id))
        
        return all_traces
    
    def visualize_semantic_consensus_graph(self,
                                           graph: nx.DiGraph,
                                           problem_type: str,
                                           max_nodes: int,
                                           output_file: str = "consensus_tree.png",
                                           font_name: str = "Nimbus Sans") -> None:
        """
        Improved visualization for reasoning-element graphs.
        - Adds a title and legend.
        - Uses consistent, categorical colors for element nodes.
        - Sequential edges define vertical flow (top → bottom).
        - Modern aesthetic with light fill and dark borders/fonts.
        """

        # --- 1. Setup & Directory ---
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if graph.number_of_nodes() == 0:
            print("Empty graph.")
            return

        # --- 2. Define Element Color Map (Consistent Coloring) ---
        all_elements = [
            'self-awareness', 'self-evaluation', 'logical-coherence', 
            'compositionality', 'sequential-organization', 'selective-attention', 
            'forward-chaining', 'causal-organization', 'knowledge-structure-alignment', 
            'strategy-selection', 'goal-management', 'ordinal-organization', 
            'temporal-organization', 'context-alignment', 'verification', 
            'backtracking', 'conceptual-level-processing', 
            'decomposition-and-integration', 'representational-restructuring', 
            'abstraction', 'backward-chaining', 'productivity', 
            'hierarchical-organization', 'adaptive-detail-management', 
            'pattern-recognition', 'spatial-organization', 'network-organization', 
            'context-awareness'
        ]
        
        cmap_tab20 = plt.cm.get_cmap("Set2")
        cmap_tab20b = plt.cm.get_cmap("Set3")
        colors_list = [cmap_tab20(i) for i in range(cmap_tab20.N)] + \
                    [cmap_tab20b(i) for i in range(cmap_tab20b.N)]
        
        element_color_map = {}
        for i, element in enumerate(all_elements):
            r, g, b, _ = colors_list[i % len(colors_list)]
            element_color_map[element] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        
        default_color = "#AAAAAA"

        # --- 3. Compute Node "Levels" (same as before) ---
        seq_edges = [(u, v) for u, v, d in graph.edges(data=True)
                    if d.get("edge_type") == "next"]

        seq_graph = nx.DiGraph()
        seq_graph.add_nodes_from(graph.nodes())
        seq_graph.add_edges_from(seq_edges)

        if seq_graph.number_of_edges() > 0:
            try:
                levels = nx.topological_generations(seq_graph)
                level_map = {}
                for depth, layer in enumerate(levels):
                    for node in layer:
                        level_map[node] = depth
            except nx.NetworkXUnfeasible:
                level_map = {n: 0 for n in graph.nodes()}
        else:
            level_map = {n: 0 for n in graph.nodes()}

        # --- 4. Use Graphviz DOT Layout ---

        current_nodes = graph.number_of_nodes()
        title_label = f"""<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="4">
        <TR><TD COLSPAN="2"><B>{problem_type} (Nodes: {current_nodes} / {max_nodes})</B></TD></TR>
        <TR><TD COLSPAN="2"><HR></HR></TD></TR>
        <TR><TD ALIGN="RIGHT">Next:</TD><TD ALIGN="LEFT"><FONT COLOR="#3498db">━━━</FONT> (solid)</TD></TR>
        <TR><TD ALIGN="RIGHT">Contains:</TD><TD ALIGN="LEFT"><FONT COLOR="#e74c3c">‑ ‑ ‑</FONT> (dashed)</TD></TR>
        <TR><TD ALIGN="RIGHT">Parallel:</TD><TD ALIGN="LEFT"><FONT COLOR="#2ecc71">. . .</FONT> (dotted)</TD></TR>
        </TABLE>
        >"""

        A = pgv.AGraph(directed=True, 
                        rankdir="TB",
                        labelloc="t",
                        label=title_label,
                        fontsize="14",
                        pad="0.5",
                        splines="spline") # Use curved lines

        if font_name in A.graph_attr['fontname']:
            A.graph_attr['fontname'] = font_name
            A.node_attr['fontname'] = font_name
            A.edge_attr['fontname'] = font_name
        else:
            pass

        
        for node in graph.nodes():
            # Get the base color
            base_color = element_color_map.get(node, default_color)
            # Generate light fill and dark border/font
            fill_color, font_and_border_color = self._get_modern_colors(base_color)

            A.add_node(
                node,
                label=node,
                shape="circle",
                style="filled",
                fillcolor=fill_color,
                color=font_and_border_color,
                fontcolor=font_and_border_color,
                penwidth="2.5",
                fontsize="12"
            )

        # Add edges
        for u, v, d in graph.edges(data=True):
            e_type = d.get("edge_type", "next")
            
            color = {"next": "#3498db",
                    "contains": "#e74c3c",
                    "parallel": "#2ecc71"}[e_type]

            style = {"next": "solid",
                    "contains": "dashed",
                    "parallel": "dotted"}[e_type]

            A.add_edge(u, v,
                    color=color,
                    penwidth="2.5",
                    style=style,
                    arrowsize="1.0")

        # Enforce ranks so sequential layers align
        layers = {}
        for n, lvl in level_map.items():
            layers.setdefault(lvl, []).append(n)

        for lvl_nodes in layers.values():
            A.add_subgraph(lvl_nodes, rank="same")

        # Save layout
        A.layout(prog="dot")
        A.draw(output_file)
        print(f"Saved graph to: {output_file}")
    
    def _get_modern_colors(self, base_hex: str):
        """
        Generates a light fill color and a dark border/font color from a 
        single base hex color.
        """
        # 1. Convert hex to RGB normalized to [0, 1]
        base_hex = base_hex.lstrip('#')
        r, g, b = [int(base_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
        
        # 2. Convert RGB to HLS (Hue, Lightness, Saturation)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        
        # 3. Create light (fill) and dark (border/font) versions
        # Light version: increase lightness (e.g., 80% towards white)
        light_l = l + (1.0 - l) * 0.8
        # Dark version: decrease lightness (e.g., 60% towards black)
        dark_l = l * 0.6
        
        # 4. Convert back to RGB
        light_r, light_g, light_b = colorsys.hls_to_rgb(h, light_l, s)
        dark_r, dark_g, dark_b = colorsys.hls_to_rgb(h, dark_l, s)
        
        # 5. Helper to convert normalized RGB back to hex
        def to_hex(nr, ng, nb):
            return f"#{int(nr*255):02x}{int(ng*255):02x}{int(nb*255):02x}"
            
        fill_color = to_hex(light_r, light_g, light_b)
        border_color = to_hex(dark_r, dark_g, dark_b)

        if dark_r > 0.95:
            border_color = "#172849" # Dark blue
        
        return fill_color, border_color