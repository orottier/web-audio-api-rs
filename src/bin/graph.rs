use std::collections::HashMap;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct NodeIndex(usize);

#[derive(Debug)]
pub struct Graph {
    increment: usize,
    nodes: HashMap<NodeIndex, AudioNode>,
    edges: HashMap<(NodeIndex, NodeIndex), Connection>,

    marked: Vec<NodeIndex>,
    ordered: Vec<NodeIndex>,
}

#[derive(Debug)]
pub struct AudioNode {
    pub name: String,
    pub buffer: String,
}

impl AudioNode {
    pub fn new(name: String) -> Self {
        Self {
            name,
            buffer: String::new(),
        }
    }

    pub fn set_buffer(&mut self, buf: String) {
        self.buffer = buf;
    }
}

#[derive(Debug)]
pub struct Connection {
    pub channel: usize,
}

impl Graph {
    pub fn new(root: AudioNode) -> Self {
        let mut graph = Graph {
            increment: 0,
            nodes: HashMap::new(),
            edges: HashMap::new(),
            ordered: vec![NodeIndex(0)],
            marked: vec![NodeIndex(0)],
        };

        graph.add_node(root);

        graph
    }

    pub fn root(&self) -> NodeIndex {
        NodeIndex(0)
    }

    pub fn add_node(&mut self, node: AudioNode) -> NodeIndex {
        let index = NodeIndex(self.increment);
        self.increment += 1;

        self.nodes.insert(index, node);

        index
    }

    pub fn add_edge(&mut self, source: NodeIndex, dest: NodeIndex, data: Connection) {
        self.edges.insert((source, dest), data);

        self.order_nodes();
    }

    pub fn children(&self, node: NodeIndex) -> impl Iterator<Item = (NodeIndex, &Connection)> {
        self.edges
            .iter()
            .filter(move |(&(_s, d), _e)| d == node)
            .map(|(&(s, _d), e)| (s, e))
    }

    pub fn children_mut(
        &mut self,
        node: NodeIndex,
    ) -> impl Iterator<Item = (NodeIndex, &mut Connection)> {
        self.edges
            .iter_mut()
            .filter(move |(&(_s, d), _e)| d == node)
            .map(|(&(s, _d), e)| (s, e))
    }

    fn visit(&self, n: NodeIndex, marked: &mut Vec<NodeIndex>, ordered: &mut Vec<NodeIndex>) {
        if marked.contains(&n) {
            return;
        }
        marked.push(n);
        self.children(n)
            .for_each(|c| self.visit(c.0, marked, ordered));
        ordered.insert(0, n);
    }

    pub fn ordered_nodes(&self) -> &[NodeIndex] {
        &self.ordered
    }

    fn order_nodes(&mut self) {
        // empty ordered_nodes, and temporarily move out of self (no allocs)
        let mut ordered = std::mem::replace(&mut self.ordered, vec![]);
        ordered.resize(self.nodes.len(), NodeIndex(0));
        ordered.clear();

        // empty marked_nodes, and temporarily move out of self (no allocs)
        let mut marked = std::mem::replace(&mut self.marked, vec![]);
        marked.resize(self.nodes.len(), NodeIndex(0));
        marked.clear();

        // start by visiting the root node
        let start = NodeIndex(0);
        self.visit(start, &mut marked, &mut ordered);

        ordered.reverse();

        // re-instate vecs to prevent new allocs
        self.ordered = ordered;
        self.marked = marked;
    }

    pub fn render(&mut self) {
        // split (mut) borrows
        let ordered = &self.ordered;
        let edges = &self.edges;
        let nodes = &mut self.nodes;

        ordered.iter().for_each(|index| {
            dbg!(("iterate ordered", index));

            // remove node from map, re-insert later (for borrowck reasons)
            let mut node = nodes.remove(index).unwrap();

            edges
                .iter()
                .filter_map(
                    move |((s, d), e)| {
                        if d == index {
                            Some((s, e))
                        } else {
                            None
                        }
                    },
                )
                .for_each(|(input_index, connection)| {
                    dbg!(("has connected", input_index));
                    let input = nodes.get(input_index).unwrap();

                    node.buffer.push_str(&input.buffer);
                    for _ in 0..connection.channel {
                        node.buffer.push_str(">");
                    }
                });

            nodes.insert(*index, node);
        });
    }
}

pub fn main() {
    let dest = AudioNode::new("dest".into());
    let mut graph = Graph::new(dest);

    let mut source = AudioNode::new("source".into());
    source.set_buffer("12345".to_string());

    let s = graph.add_node(source);
    let m = graph.add_node(AudioNode::new("mid".into()));
    dbg!((s, m));

    graph.add_edge(s, m, Connection { channel: 1 });
    graph.add_edge(s, graph.root(), Connection { channel: 1 });
    graph.add_edge(m, graph.root(), Connection { channel: 2 });

    dbg!(graph.ordered_nodes());

    graph.render();
    dbg!(graph);
}
