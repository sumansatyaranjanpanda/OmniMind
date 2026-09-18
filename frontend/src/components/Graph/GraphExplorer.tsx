import React, { useEffect, useRef, useState } from 'react';
import * as d3Force from 'd3-force';
import { select } from 'd3-selection';
import { zoom } from 'd3-zoom';
import { drag } from 'd3-drag';
import {
  Share2,
  Search,
  RotateCcw,
  Info,
  Navigation,
} from 'lucide-react';
import { KnowledgeGraphData, GraphNode } from '../../types';
import { getKnowledgeGraph, searchGraph, getNeighborhood } from '../../services/api';

const NODE_COLORS: Record<string, string> = {
  CONCEPT: '#8b5cf6',
  TECHNOLOGY: '#06b6d4',
  ORGANIZATION: '#10b981',
  DOCUMENT: '#f59e0b',
  ENTITY: '#ec4899',
};

export const GraphExplorer: React.FC = () => {
  const [graphData, setGraphData] = useState<KnowledgeGraphData>({ nodes: [], edges: [] });
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [maxHops, setMaxHops] = useState(2);

  const svgRef = useRef<SVGSVGElement | null>(null);
  const simulationRef = useRef<any>(null);

  useEffect(() => {
    loadGraph();
  }, []);

  const loadGraph = async () => {
    try {
      const data = await getKnowledgeGraph();
      setGraphData(data);
      if (data.nodes.length > 0) {
        setSelectedNode(data.nodes[0]);
      }
    } catch (err) {
      console.error('Failed to load knowledge graph:', err);
    }
  };

  useEffect(() => {
    if (!svgRef.current || graphData.nodes.length === 0) return;

    const width = svgRef.current.clientWidth || 800;
    const height = svgRef.current.clientHeight || 550;

    const svg = select(svgRef.current);
    svg.selectAll('*').remove();

    // Container for zoom/pan
    const g = svg.append('g');

    // Zoom behavior
    const zoomBehavior = zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoomBehavior as any);

    // Deep clone data for D3 mutation
    const nodes: any[] = graphData.nodes.map((d) => ({ ...d }));
    const links: any[] = graphData.edges.map((d) => ({ ...d }));

    // Force Simulation
    const simulation = d3Force
      .forceSimulation(nodes)
      .force(
        'link',
        d3Force
          .forceLink(links)
          .id((d: any) => d.id)
          .distance(120)
      )
      .force('charge', d3Force.forceManyBody().strength(-300))
      .force('center', d3Force.forceCenter(width / 2, height / 2))
      .force('collision', d3Force.forceCollide().radius(35));

    simulationRef.current = simulation;

    // Arrow markers
    svg
      .append('defs')
      .append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 24)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', 'rgba(255, 255, 255, 0.25)');

    // Render Links
    const link = g
      .append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', 'rgba(255, 255, 255, 0.15)')
      .attr('stroke-width', 1.5)
      .attr('marker-end', 'url(#arrow)');

    // Link Labels (Relation)
    const linkText = g
      .append('g')
      .selectAll('text')
      .data(links)
      .join('text')
      .text((d: any) => d.relation || '')
      .attr('font-size', '9.5px')
      .attr('font-family', 'var(--font-mono)')
      .attr('fill', 'var(--text-muted)')
      .attr('text-anchor', 'middle');

    // Render Nodes
    const dragBehavior = drag<any, any>()
      .on('start', (event: any, d: any) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', (event: any, d: any) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on('end', (event: any, d: any) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });

    const node = g
      .append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('cursor', 'pointer')
      .call(dragBehavior as any);

    // Node circles with glowing borders
    node
      .append('circle')
      .attr('r', 18)
      .attr('fill', (d: any) => NODE_COLORS[d.type] || '#8b5cf6')
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.8)
      .attr('filter', 'drop-shadow(0 0 8px rgba(139, 92, 246, 0.4))');

    // Node labels
    node
      .append('text')
      .text((d: any) => d.name)
      .attr('x', 24)
      .attr('y', 4)
      .attr('font-size', '12px')
      .attr('font-weight', '600')
      .attr('font-family', 'var(--font-sans)')
      .attr('fill', '#ffffff');

    // Click handler
    node.on('click', (_event, d: any) => {
      setSelectedNode(d);
    });

    // Tick update
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      linkText
        .attr('x', (d: any) => (d.source.x + d.target.x) / 2)
        .attr('y', (d: any) => (d.source.y + d.target.y) / 2 - 4);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    return () => {
      simulation.stop();
    };
  }, [graphData]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    try {
      const results = await searchGraph(searchQuery);
      if (results.length > 0) {
        setSelectedNode(results[0]);
      }
    } catch (err) {
      console.error('Graph search failed:', err);
    }
  };

  const handleFetchNeighborhood = async () => {
    if (!selectedNode) return;
    try {
      const neighborhood = await getNeighborhood(selectedNode.name, maxHops);
      if (neighborhood.length > 0) {
        // Highlight active sub-graph
      }
    } catch (err) {
      console.error('Neighborhood lookup failed:', err);
    }
  };

  return (
    <div
      style={{
        flex: 1,
        minHeight: 0,
        display: 'flex',
        flexDirection: 'column',
        padding: '24px 32px',
        background: 'var(--bg-space)',
        overflow: 'hidden',
      }}
    >
      {/* Top Controls Bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '16px',
          flexWrap: 'wrap',
          gap: '12px',
        }}
      >
        <div>
          <h1
            style={{
              fontFamily: 'var(--font-heading)',
              fontSize: '22px',
              fontWeight: 700,
              color: '#ffffff',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
          >
            <Share2 size={22} color="#8b5cf6" />
            <span>GraphRAG Knowledge Graph Explorer</span>
          </h1>
          <p style={{ fontSize: '12.5px', color: 'var(--text-muted)' }}>
            Force-directed entity-relationship network for multi-hop RAG discovery.
          </p>
        </div>

        {/* Search & Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ position: 'relative', width: '260px' }}>
            <input
              type="text"
              className="input-text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Search entity (e.g. 'Jina CLIP')..."
              style={{ paddingLeft: '34px', fontSize: '13px' }}
            />
            <Search
              size={15}
              color="var(--text-muted)"
              style={{ position: 'absolute', left: '12px', top: '12px' }}
            />
          </div>

          <button onClick={loadGraph} className="btn btn-secondary" title="Reset Graph Layout">
            <RotateCcw size={15} />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Main Grid: D3 Network Canvas + Node Inspector Drawer */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '16px', flex: 1 }}>
        {/* D3 Force Canvas Panel */}
        <div
          className="glass-panel"
          style={{
            position: 'relative',
            overflow: 'hidden',
            borderRadius: 'var(--radius-lg)',
            border: '1px solid var(--border-subtle)',
            background: 'var(--bg-deep)',
          }}
        >
          <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />

          {/* Legend Overlay */}
          <div
            style={{
              position: 'absolute',
              bottom: '16px',
              left: '16px',
              background: 'var(--bg-glass-heavy)',
                            padding: '10px 14px',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border-subtle)',
              display: 'flex',
              gap: '14px',
              fontSize: '11.5px',
            }}
          >
            {Object.entries(NODE_COLORS).map(([type, color]) => (
              <div key={type} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span
                  style={{
                    width: '10px',
                    height: '10px',
                    borderRadius: '50%',
                    backgroundColor: color,
                    display: 'inline-block',
                  }}
                />
                <span style={{ color: 'var(--text-secondary)' }}>{type}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Selected Entity Inspector */}
        <div
          className="glass-panel"
          style={{
            padding: '20px',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            background: 'var(--bg-surface)',
          }}
        >
          <div>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                marginBottom: '14px',
                paddingBottom: '12px',
                borderBottom: '1px solid var(--border-subtle)',
              }}
            >
              <Info size={18} color="#8b5cf6" />
              <h3 style={{ fontSize: '15px', fontWeight: 600, color: '#ffffff' }}>
                Entity Inspector
              </h3>
            </div>

            {selectedNode ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                    Entity Name
                  </div>
                  <div style={{ fontSize: '18px', fontWeight: 700, color: '#ffffff' }}>
                    {selectedNode.name}
                  </div>
                </div>

                <div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                    Entity Classification
                  </div>
                  <span
                    className="badge"
                    style={{
                      background: 'rgba(139, 92, 246, 0.12)',
                      color: NODE_COLORS[selectedNode.type] || '#a78bfa',
                      border: '1px solid rgba(139, 92, 246, 0.3)',
                      marginTop: '4px',
                    }}
                  >
                    {selectedNode.type}
                  </span>
                </div>

                <div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                    Summary / Semantic Context
                  </div>
                  <div
                    style={{
                      fontSize: '13px',
                      color: 'var(--text-secondary)',
                      lineHeight: 1.5,
                      marginTop: '4px',
                      background: 'var(--bg-deep)',
                      padding: '10px',
                      borderRadius: '6px',
                      border: '1px solid var(--border-subtle)',
                    }}
                  >
                    {selectedNode.summary || 'Core node extracted during GraphRAG discovery.'}
                  </div>
                </div>

                {/* Subgraph Hop Explorer */}
                <div style={{ marginTop: '8px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '6px' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Ego Hop Distance:</span>
                    <span style={{ color: '#a78bfa', fontWeight: 600 }}>{maxHops} Hops</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={3}
                    value={maxHops}
                    onChange={(e) => setMaxHops(parseInt(e.target.value))}
                    style={{ width: '100%', accentColor: '#8b5cf6' }}
                  />
                  <button
                    onClick={handleFetchNeighborhood}
                    className="btn btn-secondary"
                    style={{ width: '100%', marginTop: '8px', fontSize: '12px' }}
                  >
                    <Navigation size={13} />
                    <span>Extract Subgraph</span>
                  </button>
                </div>
              </div>
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: '13px' }}>
                Click any node in the graph to inspect relationships and multi-hop paths.
              </div>
            )}
          </div>

          <div
            style={{
              fontSize: '11.5px',
              color: 'var(--text-muted)',
              borderTop: '1px solid var(--border-subtle)',
              paddingTop: '12px',
            }}
          >
            NetworkX MultiDiGraph • Tenant-Isolated
          </div>
        </div>
      </div>
    </div>
  );
};
