from src.vector_search import SemanticVectorSearch
import json

def interactive_search():
    # Load results
    print("📄 Loading analysis results...")
    with open('outputs/results/complete_analysis.json', 'r') as f:
        data = json.load(f)
    
    results = data['semantic_results']
    
    # Build search index
    print("🔍 Building semantic search index...")
    searcher = SemanticVectorSearch()
    searcher.build_vector_index(results)
    
    print("\n🎯 SEMANTIC SEARCH SYSTEM READY!")
    print("=" * 50)
    print("💡 Try queries like:")
    print("  - 'COVID-19 symptoms and diagnosis'")
    print("  - 'antiviral treatments'") 
    print("  - 'vaccine development'")
    print("  - 'hospital protocols'")
    print("=" * 50)
    
    while True:
        query = input("\n🔎 Enter your search query (or 'quit' to exit): ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not query.strip():
            continue
            
        # Perform search
        print(f"🔍 Searching for: '{query}'...")
        search_results = searcher.search(query, top_k=5)
        
        if not search_results:
            print("❌ No results found.")
            continue
            
        print(f"\n📋 Found {len(search_results)} results:")
        print("-" * 60)
        
        for i, result in enumerate(search_results, 1):
            print(f"\n{i}. 📄 {result['title']}")
            print(f"   📊 Similarity Score: {result['similarity_score']:.3f}")
            print(f"   🆔 Paper ID: {result['paper_id']}")
            print(f"   🏷️ Entities: {result['entity_count']} | Relations: {result['relation_count']}")
            
            # Show a snippet of text if available
            if 'text' in result and result['text']:
                snippet = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
                print(f"   📝 Preview: {snippet}")
            
    print("\n👋 Thanks for using CORD-19 Semantic Search!")

if __name__ == "__main__":
    interactive_search()