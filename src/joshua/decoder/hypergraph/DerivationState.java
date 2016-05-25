package joshua.decoder.hypergraph;

import java.util.List;

import joshua.corpus.Vocabulary;
import joshua.decoder.ff.FeatureVector;
import joshua.decoder.ff.tm.Rule;
import joshua.decoder.hypergraph.KBestExtractor.DerivationVisitor;
import joshua.decoder.hypergraph.KBestExtractor.Side;
import joshua.decoder.hypergraph.KBestExtractor.VirtualNode;

/**
 * A DerivationState describes which path to follow through the hypergraph. For example, it
 * might say to use the 1-best from the first tail node, the 9th-best from the second tail node,
 * and so on. This information is represented recursively through a chain of DerivationState
 * objects. This function follows that chain, extracting the information according to a number
 * of parameters, and returning results to a string, and also (optionally) accumulating the
 * feature values into the passed-in FeatureVector.
 */

// each DerivationState roughly corresponds to a hypothesis
public class DerivationState {
  /**
   * 
   */
  private final KBestExtractor kBestExtractor;

  /* The edge ("e" in the paper) */
  public HyperEdge edge;

  /* The edge's parent node */
  public HGNode parentNode;

  /*
   * This state's position in its parent node's list of incoming hyperedges (used in signature
   * calculation)
   */
  public int edgePos;

  /*
   * The rank item to select from each of the incoming tail nodes ("j" in the paper, an ArrayList
   * of size |e|)
   */
  public int[] ranks;

  /*
   * The cost of the hypothesis, including a weighted BLEU score, if any.
   */
  private float cost;

  public DerivationState(KBestExtractor kBestExtractor, HGNode pa, HyperEdge e, int[] r, float c, int pos) {
    this.kBestExtractor = kBestExtractor;
    parentNode = pa;
    edge = e;
    ranks = r;
    cost = c;
    edgePos = pos;
  }

  public void setCost(float cost2) {
    this.cost = cost2;
  }

  /**
   * Returns the model cost.
   * 
   * @return
   */
  public float getModelCost() {
    return this.cost;
  }

  /**
   * Returns the model cost;
   * 
   * @return
   */
  public float getCost() {
    return cost;
  }

  public String toString() {
    StringBuilder sb = new StringBuilder(String.format("DS[[ %s (%d,%d)/%d ||| ",
        Vocabulary.word(parentNode.lhs), parentNode.i, parentNode.j, edgePos));
    sb.append("ranks=[ ");
    if (ranks != null)
      for (int i = 0; i < ranks.length; i++)
        sb.append(ranks[i] + " ");
    sb.append("] ||| " + String.format("%.5f ]]", cost));
    return sb.toString();
  }

  public boolean equals(Object other) {
    if (other instanceof DerivationState) {
      DerivationState that = (DerivationState) other;
      if (edgePos == that.edgePos) {
        if (ranks != null && that.ranks != null) {
          if (ranks.length == that.ranks.length) {
            for (int i = 0; i < ranks.length; i++)
              if (ranks[i] != that.ranks[i])
                return false;
            return true;
          }
        }
      }
    }

    return false;
  }

  /**
   * DerivationState objects are unique to each VirtualNode, so the unique identifying information
   * only need contain the edge position and the ranks.
   */
  public int hashCode() {
    int hash = edgePos;
    if (ranks != null) {
      for (int i = 0; i < ranks.length; i++)
        hash = hash * 53 + i;
    }

    return hash;
  }

  /**
   * Visits every state in the derivation in a depth-first order.
   */
  private DerivationVisitor visit(DerivationVisitor visitor) {
    return visit(visitor, 0, 0);
  }

  private DerivationVisitor visit(DerivationVisitor visitor, int indent, int tailNodeIndex) {

    visitor.before(this, indent, tailNodeIndex);

    final Rule rule = edge.getRule();
    final List<HGNode> tailNodes = edge.getTailNodes();

    if (rule == null) {
      /* A null rule is a shortcut for the top level of the derivation tree */
      getChildDerivationState(edge, 0).visit(visitor, indent + 1, 0);
    } else {
      if (tailNodes != null) {
        for (int index = 0; index < tailNodes.size(); index++) {
          getChildDerivationState(edge, index).visit(visitor, indent + 1, index);
        }
      }
    }

    visitor.after(this, indent, tailNodeIndex);

    return visitor;
  }
  
  public WordAlignmentState getWordAlignment() {
    WordAlignmentExtractor extractor = new WordAlignmentExtractor();
    visit(extractor);
    return extractor.getFinalWordAlignments();
  }

  public String getTree() {
    return visit(this.kBestExtractor.new TreeExtractor()).toString();
  }
  
  public String getHypothesis() {
    return getHypothesis(this.kBestExtractor.defaultSide);
  }

  /**
   * For stack decoding we keep using the old string-based
   * HypothesisExtractor.
   * For Hiero, we use a faster, int-based hypothesis extraction
   * that is correct also for Side.SOURCE cases.
   */
  public String getHypothesis(final Side side) {
    return visit(new OutputStringExtractor(side.equals(Side.SOURCE))).toString();
  }

  public FeatureVector getFeatures() {
    final FeatureVectorExtractor extractor = new FeatureVectorExtractor(this.kBestExtractor.featureFunctions, this.kBestExtractor.sentence);
    visit(extractor);
    return extractor.getFeatures();
  }

  public String getDerivation() {
    return visit(this.kBestExtractor.new DerivationExtractor()).toString();
  }

  /**
   * Helper function for navigating the hierarchical list of DerivationState objects. This
   * function looks up the VirtualNode corresponding to the HGNode pointed to by the edge's
   * {tailNodeIndex}th tail node.
   * 
   * @param edge
   * @param tailNodeIndex
   * @return
   */
  public DerivationState getChildDerivationState(HyperEdge edge, int tailNodeIndex) {
    HGNode child = edge.getTailNodes().get(tailNodeIndex);
    VirtualNode virtualChild = this.kBestExtractor.getVirtualNode(child);
    return virtualChild.nbests.get(ranks[tailNodeIndex] - 1);
  }

} // end of Class DerivationState