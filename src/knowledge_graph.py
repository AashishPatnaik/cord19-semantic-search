import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict
import pandas as pd
from collections import Counter
import json

class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()
        self.entity_colors = {
            'PERSON': '#FF6B6B',
            'ORG': '#4ECDC4', 
            'GPE': '#45B7D1',
            'PRODUCT': '#96CEB4',
            'DISEASE': '#FFEAA7',
            'CHEMICAL': '#DDA0DD'
        }
    
    def build_graph(self, semantic_results: List[Dict]) -> nx.Graph:
        """Build knowledge graph from semantic extraction results"""
        
        print("🕸️ Building knowledge graph...")
        
        # Add entity nodes
        all_entities = []
        for paper in semantic_results:
            for entity in paper.get('entities', []):
                all_entities.append(entity)
        
        # Count entity frequencies
        entity_counts = Counter([(e['text'], e['label']) for e in all_entities])
        
        # Add nodes (entities)
        for (entity_text, entity_type), count in entity_counts.items():
            self.graph.add_node(
                entity_text,
                type=entity_type,
                frequency=count,
                size=min(count * 10, 100)  # Node size based on frequency
            )
        
        # Add relationship edges
        for paper in semantic_results:
            for relation in paper.get('relations', []):
                subject = relation['subject']
                obj = relation['object']
                predicate = relation['predicate']
                
                if self.graph.has_node(subject) and self.graph.has_node(obj):
                    if self.graph.has_edge(subject, obj):
                        # Increase edge weight if relation already exists
                        self.graph[subject][obj]['weight'] += 1
                        self.graph[subject][obj]['relations'].append(predicate)
                    else:
                        # Add new edge
                        self.graph.add_edge(
                            subject, obj,
                            weight=1,
                            relations=[predicate],
                            confidence=relation.get('confidence', 0.5)
                        )
        
        print(f"✅ Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph
    
    def create_interactive_graph(self, output_file='outputs/results/knowledge_graph.html'):
        """Create interactive Plotly visualization"""
        
        if self.graph.number_of_nodes() == 0:
            print("❌ No graph to visualize!")
            return
        
        # Use spring layout for positioning
        pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # Prepare node data
        node_trace = go.Scatter(
            x=[], y=[], text=[], mode='markers+text',
            hoverinfo='text',
            marker=dict(size=[], color=[], line=dict(width=2))
        )
        
        # Add nodes
        for node in self.graph.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
            
            node_info = self.graph.nodes[node]
            entity_type = node_info.get('type', 'UNKNOWN')
            frequency = node_info.get('frequency', 1)
            
            node_trace['text'] += (node,)
            node_trace['marker']['size'] += (min(frequency * 15, 50),)
            node_trace['marker']['color'] += (self.entity_colors.get(entity_type, '#999999'),)
        
        # Prepare edge data
        edge_trace = go.Scatter(x=[], y=[], mode='lines', line=dict(width=1, color='#888'))
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=dict(
                           title='CORD-19 Knowledge Graph',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Knowledge Graph from COVID-19 Research Papers",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        fig.write_html(output_file)
        print(f"📊 Interactive graph saved: {output_file}")
    
    def analyze_graph(self) -> Dict:
        """Analyze graph properties"""
        
        if self.graph.number_of_nodes() == 0:
            return {}
        
        analysis = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'top_entities': {},
            'top_connections': {},
            'entity_types': {}
        }
        
        # Top entities by connections
        degrees = dict(self.graph.degree())
        top_entities = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        analysis['top_entities'] = dict(top_entities)
        
        # Entity type distribution
        for node in self.graph.nodes(data=True):
            entity_type = node[1].get('type', 'UNKNOWN')
            analysis['entity_types'][entity_type] = analysis['entity_types'].get(entity_type, 0) + 1
        
        return analysis