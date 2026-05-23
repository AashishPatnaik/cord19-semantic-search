import  json
import os
import glob
import sys
sys.path.append('src')

from src.ner_extractor import SimpleNER
from src.relation_extractor import SimpleRE
from src.knowledge_graph import KnowledgeGraphBuilder
from src.vector_search import SemanticVectorSearch

def main():
    print("🚀 Starting Enhanced CORD-19 Processing...")
    
    # Load papers
    paper_files = glob.glob('data/papers/*.json')
    print(f"Found {len(paper_files)} papers")
    
    # Initialize all processors
    ner = SimpleNER()
    re_extractor = SimpleRE()
    kg_builder = KnowledgeGraphBuilder()
    vector_search = SemanticVectorSearch()
    
    results = []
    
    # Process papers
    for i, file_path in enumerate(paper_files):
        print(f"Processing paper {i+1}/{len(paper_files)}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                paper = json.load(f)
            
            # Extract entities
            ner_result = ner.extract_entities(paper)
            
            # Extract relations
            full_text = ner_result.get('title', '') + ' ' + ' '.join([e['text'] for e in ner_result['entities']])
            relations = re_extractor.extract_relations(full_text, ner_result['entities'])
            
            # Combine results
            ner_result['relations'] = relations
            ner_result['relation_count'] = len(relations)
            results.append(ner_result)
            
        except Exception as e:
            print(f"Error with {file_path}: {e}")
    
    # Build Knowledge Graph
    print("\n🕸️ Building Knowledge Graph...")
    kg_builder.build_graph(results)
    kg_builder.create_interactive_graph()
    graph_analysis = kg_builder.analyze_graph()
    
    # Build Vector Search Index
    print("\n🔍 Building Vector Search...")
    vector_search.build_vector_index(results)
    
    # Demo searches
    print("\n🔎 Demo Searches:")
    demo_queries = ["COVID-19 treatment", "vaccine research", "coronavirus symptoms"]
    
    search_results = {}
    for query in demo_queries:
        search_results[query] = vector_search.search(query, top_k=3)
        print(f"\nQuery: '{query}'")
        for result in search_results[query]:
            print(f"  • {result['title'][:60]}... (Score: {result['similarity_score']:.3f})")
    
    # Save everything
    print("\n💾 Saving results...")
    os.makedirs('outputs/results', exist_ok=True)
    
    # Save main results
    output_data = {
        'semantic_results': results,
        'graph_analysis': graph_analysis,
        'demo_searches': search_results,
        'summary': {
            'total_papers': len(results),
            'total_entities': sum(r['entity_count'] for r in results),
            'total_relations': sum(r['relation_count'] for r in results),
            'graph_nodes': graph_analysis.get('nodes', 0),
            'graph_edges': graph_analysis.get('edges', 0)
        }
    }
    
    with open('outputs/results/complete_analysis.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print final summary
    summary = output_data['summary']
    print(f"\n✅ COMPLETE ANALYSIS FINISHED!")
    print(f"📄 Papers processed: {summary['total_papers']}")
    print(f"🏷️ Total entities: {summary['total_entities']}")
    print(f"🔗 Total relations: {summary['total_relations']}")
    print(f"🕸️ Knowledge graph: {summary['graph_nodes']} nodes, {summary['graph_edges']} edges")
    print(f"📊 Interactive graph: outputs/results/knowledge_graph.html")
    print(f"💾 Complete results: outputs/results/complete_analysis.json")
    
    print(f"\n🎯 READY TO DEMO:")
    print(f"1. Open outputs/results/knowledge_graph.html in browser")
    print(f"2. Check outputs/results/complete_analysis.json for all data")
    print(f"3. Your semantic search system is ready!")

if __name__ == "__main__":
    main()