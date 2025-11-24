import os
import sys
from collections import defaultdict
import argparse
from structure import SpanTree, ConsensusTreeFinder

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
    parser.add_argument("--overlap_threshold", type=float, default=0.5, help="Overlap threshold for span tree")
    parser.add_argument("--parallel_threshold", type=float, default=0.5, help="Parallel threshold for span tree")
    parser.add_argument("--target_type", type=str, default=None, help="Target type to filter elements")
    parser.add_argument("--output_dir", type=str, default="output_consensus_graphs", help="Output directory for consensus graphs")
    parser.add_argument("--max_nodes", type=int, default=7, help="Maximum number of nodes in consensus graph")
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

    # Find consensus trees for each target problem type

    consensus_finder = ConsensusTreeFinder(span_tree=tree)
    type2graph = defaultdict(lambda: defaultdict(lambda: defaultdict()))

    # Generate consensus graphs for all target problem types
    problem_types = list(id2type.values())
    if "No Type Matched" in problem_types:
        problem_types.remove("No Type Matched")

    target_problem_types = problem_types[1:-2]
    target_problem_types.remove('Strategic Performance')

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