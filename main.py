import os
import sys
from collections import defaultdict
import argparse
from reasoning_structure.structure import SpanTree, ConsensusTreeFinder
from structure_guidance.generate_steered_traces import ElementGuidedReasoning

id2type = {
    -1: "No Type Matched",
    10: "Logical",
    1: "Algorithmic",
    2: "Story Problem",
    3: "Rule-Using", 
    4: "Decision-Making",
    5: "Troubleshooting",
    6: "Diagnosis-Solution",
    7: "Strategic Performance",
    8: "Case Analysis",
    9: "Design",
    11: "Dilemma",
    12: "Factual Recall/Comprehension",
    13: "Creative/Expressive"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--element_dir", type=str, default='/shared/data3/pk36/structured_survey/span_annotation', help="Directory containing element files")
    parser.add_argument("--prompt_template_dir", type=str, default='structure_guidance/prompt_templates', help="Directory containing outputted templates")
    parser.add_argument("--overlap_threshold", type=float, default=0.8, help="Overlap threshold for span tree")
    parser.add_argument("--parallel_threshold", type=float, default=20, help="Parallel threshold for span tree")
    parser.add_argument("--target_type", type=str, default=None, help="Target type to filter elements")
    parser.add_argument("--output_dir", type=str, default="output_consensus_graphs", help="Output directory for consensus graphs")
    parser.add_argument("--max_nodes", type=int, default=7, help="Maximum number of nodes in consensus graph")
    parser.add_argument("--generate_steered_traces", action="store_true")
    args = parser.parse_args()

    tree = SpanTree(
        overlap_threshold=args.overlap_threshold,
        parallel_threshold=args.parallel_threshold
    )

    # Load span-level annotation files

    if not os.path.exists(args.element_dir):
        print(f"Element directory with span-level annotations ({args.element_dir}) does not exist.")
        sys.exit(1)

    tree.load_element_files(args.element_dir, id2type=id2type, target_type=args.target_type)

    print("Data loaded successfully!")

    # Find consensus and success-prone subgraphs for each target problem type

    consensus_finder = ConsensusTreeFinder(span_tree=tree)
    type2graph = defaultdict(lambda: defaultdict(lambda: defaultdict()))

    # Generate consensus and success-prone subgraphs for all target problem types
    problem_types = list(id2type.values())
    if "No Type Matched" in problem_types:
        problem_types.remove("No Type Matched")

    target_problem_types = problem_types[1:-2]
    target_problem_types.remove('Strategic Performance') # not applicable for non-real-world settings

    success_only = True
    success_suffix = 'success' if success_only else 'all'

    for p_type in target_problem_types:
        print(f"Processing {p_type}...")
        core_graph = consensus_finder.construct_consensus_tree(p_type, max_nodes=args.max_nodes, success_only=success_only, dynamic=True)
        consensus_finder.visualize_semantic_consensus_graph(graph=core_graph, problem_type=p_type, max_nodes=args.max_nodes, 
                                                            output_file=f'{args.output_dir}/{args.max_nodes}/{success_suffix}_graph_{p_type}_{args.max_nodes}.png')
        type2graph[p_type][args.max_nodes][success_suffix] = core_graph

    success_only = False
    success_suffix = 'success' if success_only else 'all'

    for p_type in target_problem_types:
        print(f"Processing {p_type}...")
        core_graph = consensus_finder.construct_consensus_tree(p_type, max_nodes=args.max_nodes, success_only=success_only, dynamic=True)
        consensus_finder.visualize_semantic_consensus_graph(graph=core_graph, problem_type=p_type, max_nodes=args.max_nodes, 
                                                            output_file=f'{args.output_dir}/{args.max_nodes}/{success_suffix}_graph_{p_type}_{args.max_nodes}.png')
        type2graph[p_type][args.max_nodes][success_suffix] = core_graph

    print("Consensus + success-prone graphs all generated and saved successfully!")

    ## Linearize each graph
    guided_reasoning = ElementGuidedReasoning(args=args, span_tree=tree, consensus_finder=consensus_finder, problem_type_graphs=type2graph)

    type2linear = defaultdict(lambda: defaultdict()) # problem_type -> max_nodes : template
    for p_type, graphs in type2graph.items():
        for max_nodes, graph in graphs.items():
            type2linear[p_type][max_nodes] = guided_reasoning.graph_to_prompt(graph)
    
    with open('structure_guidance/template_prompt.txt', 'r') as f:
        prompt_template = f.read()

    ## Convert linearized graph into prompt (using structure_guidance/template_prompt.txt)
    for p_type in type2linear:
        for node_num in type2linear[p_type]:
            with open(f'{args.prompt_template_dir}/{p_type}_{node_num}.txt', 'w') as f:
                prompt_text = prompt_template.format(p_type=p_type, node_num=node_num, structure_info=type2linear[p_type][node_num])