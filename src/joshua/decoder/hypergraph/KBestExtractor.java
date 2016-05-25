/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package joshua.decoder.hypergraph;

import java.util.Arrays;
import java.util.Comparator;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.PriorityQueue;

import joshua.corpus.Vocabulary;
import joshua.decoder.JoshuaConfiguration;
import joshua.decoder.ff.FeatureFunction;
import joshua.decoder.ff.FeatureVector;
import joshua.decoder.ff.fragmentlm.Tree;
import joshua.decoder.ff.state_maintenance.DPState;
import joshua.decoder.ff.tm.Rule;
import joshua.decoder.segment_file.Sentence;

/**
 * This class implements lazy k-best extraction on a hyper-graph.
 * 
 * K-best extraction over hypergraphs is a little hairy, but is best understood in the following
 * manner. Imagine a hypergraph, which is composed of nodes connected by hyperedges. A hyperedge has
 * exactly one parent node and 1 or more tail nodes, corresponding to the rank of the rule that gave
 * rise to the hyperedge. Each node has 1 or more incoming hyperedges.
 * 
 * K-best extraction works in the following manner. A derivation is a set of nodes and hyperedges
 * that leads from the root node down and exactly covers the source-side sentence. To define a
 * derivation, we start at the root node, choose one of its incoming hyperedges, and then recurse to
 * the tail (or antecedent) nodes of that hyperedge, where we continually make the same decision.
 * 
 * Each hypernode has its hyperedges sorted according to their model score. To get the best
 * (Viterbi) derivation, we simply recursively follow the best hyperedge coming in to each
 * hypernode.
 * 
 * How do we get the second-best derivation? It is defined by changing exactly one of the decisions
 * about which hyperedge to follow in the recursion. Somewhere, we take the second-best. Similarly,
 * the third-best derivation makes a single change from the second-best: either making another
 * (differnt) second-best choice somewhere along the 1-best derivation, or taking the third-best
 * choice at the same spot where the second-best derivation took the second-best choice. And so on.
 * 
 * This class uses two classes that encode the necessary meta-information. The first is the
 * DerivationState class. It roughly corresponds to a hyperedge, and records, for each of that
 * hyperedge's tail nodes, which-best to take. So for a hyperedge with three tail nodes, the 1-best
 * derivation will be (1,1,1), the second-best will be one of (2,1,1), (1,2,1), or (1,1,2), the
 * third best will be one of
 * 
 * (3,1,1), (2,2,1), (1,1,3)
 * 
 * and so on.
 * 
 * The configuration parameter `output-format` controls what exactly is extracted from the forest.
 * See documentation for that below. Note that Joshua does not store individual feature values while 
 * decoding, but only the cost of each edge (in the form of a float). Therefore, if you request
 * the features values (`%f` in `output-format`), the feature functions must be replayed, which
 * is expensive.
 * 
 * The configuration parameter `top-n` controls how many items are returned. If this is set to 0,
 * k-best extraction should be turned off entirely.
 * 
 * You can call getViterbiDerivation() essentially for free. But as soon as you call hasNext()
 * (or next(), e.g., via the iterator), you're going to trigger some relatively expensive
 * k-best computation.
 * 
 * @author Zhifei Li, <zhifei.work@gmail.com>
 * @author Matt Post <post@cs.jhu.edu>
 */
public class KBestExtractor implements Iterator<DerivationState>, Iterable<DerivationState> { 
  private final JoshuaConfiguration joshuaConfiguration;
  private final HashMap<HGNode, VirtualNode> virtualNodesTable = new HashMap<HGNode, VirtualNode>();

  // static final String rootSym = JoshuaConfiguration.goal_symbol;
  static final String rootSym = "ROOT";
  static final int rootID = Vocabulary.id(rootSym);

  public enum Side {
    SOURCE, TARGET
  };

  /* Whether to extract only unique strings */
  private final boolean extractUniqueNbest;

  /* Which side to output (source or target) */
  final Side defaultSide;

  /* The input sentence */
  final Sentence sentence;

  /* The weights being used to score the forest */
  final FeatureVector weights;

  /* The feature functions */
  final List<FeatureFunction> featureFunctions;
  private HyperGraph hyperGraph;
  private DerivationState nextDerivation = null;
  private int derivationCounter;
  private int maxDerivations;

  public KBestExtractor(
      Sentence sentence,
      HyperGraph hyperGraph,
      List<FeatureFunction> featureFunctions,
      FeatureVector weights,
      boolean isMonolingual,
      JoshuaConfiguration joshuaConfiguration,
      int k) {

    this.featureFunctions = featureFunctions;
    this.hyperGraph = hyperGraph;
    this.joshuaConfiguration = joshuaConfiguration;
    this.extractUniqueNbest = joshuaConfiguration.use_unique_nbest;

    this.weights = weights;
    this.defaultSide = (isMonolingual ? Side.SOURCE : Side.TARGET);
    this.sentence = sentence;
    
    // initialize the iterator
    this.derivationCounter = 0;
    this.nextDerivation = null;
    this.maxDerivations = k;
  }

  /**
   * Returns the Viterbi derivation. You don't want to use the general k-best extraction code because
   * (a) the Viterbi derivation is always needed and (b) k-best extraction is slow. So this is basically
   * a convenience function that by-passes the expensive k-best extraction for a common use-case.
   * 
   * @return the Viterib derivation
   */
  public DerivationState getViterbiDerivation() {
    
    /* TODO: the viterbi derivation is often needed, but triggering all the k-best mechanisms
     * to extract it is expensive. There should be a way to get the 1-best DerivationState object
     * very quickly, so that we can fit it into this framework. 
     */
    throw new RuntimeException("Not yet implemented! We need a fast way to get the Viterbi DerivationState!");
  }

  
  /**
   * Compute the string that is output from the decoder, using the "output-format" config file
   * parameter as a template.
   * 
   * You may need to reset_state() before you call this function for the first time.
   */
  public DerivationState getKthHyp(HGNode node, int k) {

    // Determine the k-best hypotheses at each HGNode
    VirtualNode virtualNode = getVirtualNode(node);
    DerivationState derivationState = virtualNode.lazyKBestExtractOnNode(this, k);

    return derivationState;
  }

  // =========================== end kbestHypergraph

  /**
   * This clears the virtualNodesTable, which maintains a list of virtual nodes. This should be
   * called in between forest rescorings.
   */
  public void resetState() {
    virtualNodesTable.clear();
  }

  /**
   * Returns the VirtualNode corresponding to an HGNode. If no such VirtualNode exists, it is
   * created.
   * 
   * @param hgnode
   * @return the corresponding VirtualNode
   */
  VirtualNode getVirtualNode(HGNode hgnode) {
    VirtualNode virtualNode = virtualNodesTable.get(hgnode);
    if (null == virtualNode) {
      virtualNode = new VirtualNode(hgnode);
      virtualNodesTable.put(hgnode, virtualNode);
    }
    return virtualNode;
  }

  /**
   * This class is essentially a wrapper around an HGNode, annotating it with information needed to
   * record which hypotheses have been explored from this point. There is one virtual node for
   * each HGNode in the underlying hypergraph. This VirtualNode maintains information about the
   * k-best derivations from that point on, retaining the derivations computed so far and a priority 
   * queue of candidates.
   */

  class VirtualNode {

    // The node being annotated.
    HGNode node = null;

    // sorted ArrayList of DerivationState, in the paper is: D(^) [v]
    public List<DerivationState> nbests = new ArrayList<DerivationState>();

    // remember frontier states, best-first; in the paper, it is called cand[v]
    private PriorityQueue<DerivationState> candHeap = null;

    // Remember which DerivationState has been explored (positions in the hypercube). This allows
    // us to avoid duplicated states that are reached from different places of expansion, e.g.,
    // position (2,2) can be reached be extending (1,2) and (2,1).
    private HashSet<DerivationState> derivationTable = null;

    // This records unique *strings* at each item, used for unique-nbest-string extraction.
    private HashSet<String> uniqueStringsTable = null;

    public VirtualNode(HGNode it) {
      this.node = it;
    }

    /**
     * This returns a DerivationState corresponding to the kth-best derivation rooted at this node.
     * 
     * @param kbestExtractor
     * @param k (indexed from one)
     * @return the k-th best (1-indexed) hypothesis, or null if there are no more.
     */
    // return: the k-th hyp or null; k is started from one
    private DerivationState lazyKBestExtractOnNode(KBestExtractor kbestExtractor, int k) {
      if (nbests.size() >= k) { // no need to continue
        return nbests.get(k - 1);
      }

      // ### we need to fill in the l_nest in order to get k-th hyp
      DerivationState derivationState = null;

      /*
       * The first time this is called, the heap of candidates (the frontier of the cube) is
       * uninitialized. This recursive call will seed the candidates at each node.
       */
      if (null == candHeap) {
        getCandidates(kbestExtractor);
      }

      /*
       * Now build the kbest list by repeatedly popping the best candidate and then placing all
       * extensions of that hypothesis back on the candidates list.
       */
      int tAdded = 0; // sanity check
      while (nbests.size() < k) {
        if (candHeap.size() > 0) {
          derivationState = candHeap.poll();
          // derivation_tbl.remove(res.get_signature());//TODO: should remove? note that two state
          // may be tied because the cost is the same
          if (extractUniqueNbest) {
            // We pass false for extract_nbest_tree because we want; to check that the hypothesis
            // *strings* are unique, not the trees.
            final String res_str = derivationState.getHypothesis();
            
            if (!uniqueStringsTable.contains(res_str)) {
              nbests.add(derivationState);
              uniqueStringsTable.add(res_str);
            }
          } else {
            nbests.add(derivationState);
          }

          // Add all extensions of this hypothesis to the candidates list.
          lazyNext(kbestExtractor, derivationState);

          // debug: sanity check
          tAdded++;
          // this is possible only when extracting unique nbest
          if (!extractUniqueNbest && tAdded > 1) {
            throw new RuntimeException("In lazyKBestExtractOnNode, add more than one time, k is "
                + k);
          }
        } else {
          break;
        }
      }
      if (nbests.size() < k) {
        derivationState = null;// in case we do not get to the depth of k
      }
      // debug: sanity check
      // if (l_nbest.size() >= k && l_nbest.get(k-1) != res) {
      // throw new RuntimeException("In lazy_k_best_extract, ranking is not correct ");
      // }

      return derivationState;
    }

    /**
     * This function extends the current hypothesis, adding each extended item to the list of
     * candidates (assuming they have not been added before). It does this by, in turn, extending
     * each of the tail node items.
     * 
     * @param kbestExtractor
     * @param previousState
     */
    private void lazyNext(KBestExtractor kbestExtractor, DerivationState previousState) {
      /* If there are no tail nodes, there is nothing to do. */
      if (null == previousState.edge.getTailNodes())
        return;

      /* For each tail node, create a new state candidate by "sliding" that item one position. */
      for (int i = 0; i < previousState.edge.getTailNodes().size(); i++) {
        /* Create a new virtual node that is a copy of the current node */
        HGNode tailNode = (HGNode) previousState.edge.getTailNodes().get(i);
        VirtualNode virtualTailNode = kbestExtractor.getVirtualNode(tailNode);
        // Copy over the ranks.
        int[] newRanks = new int[previousState.ranks.length];
        for (int c = 0; c < newRanks.length; c++) {
          newRanks[c] = previousState.ranks[c];
        }
        // Now increment/slide the current tail node by one
        newRanks[i] = previousState.ranks[i] + 1;

        // Create a new state so we can see if it's new. The cost will be set below if it is.
        DerivationState nextState = new DerivationState(KBestExtractor.this, previousState.parentNode,
            previousState.edge, newRanks, 0.0f, previousState.edgePos);

        // Don't add the state to the list of candidates if it's already been added.
        if (!derivationTable.contains(nextState)) {
          // Make sure that next candidate exists
          virtualTailNode.lazyKBestExtractOnNode(kbestExtractor, newRanks[i]);
          // System.err.println(String.format("  newRanks[%d] = %d and tail size %d", i,
          // newRanks[i], virtualTailNode.nbests.size()));
          if (newRanks[i] <= virtualTailNode.nbests.size()) {
            // System.err.println("NODE: " + this.node);
            // System.err.println("  tail is " + virtualTailNode.node);
            float cost = previousState.getModelCost()
                - virtualTailNode.nbests.get(previousState.ranks[i] - 1).getModelCost()
                + virtualTailNode.nbests.get(newRanks[i] - 1).getModelCost();
            nextState.setCost(cost);

            candHeap.add(nextState);
            derivationTable.add(nextState);

            // System.err.println(String.format("  LAZYNEXT(%s", nextState));
          }
        }
      }
    }

    /**
     * this is the seeding function, for example, it will get down to the leaf, and sort the
     * terminals get a 1best from each hyperedge, and add them into the heap_cands
     * 
     * @param kbestExtractor
     */
    private void getCandidates(KBestExtractor kbestExtractor) {
      /* The list of candidates extending from this (virtual) node. */
      candHeap = new PriorityQueue<DerivationState>(11, new DerivationStateComparator());

      /*
       * When exploring the cube frontier, there are multiple paths to each candidate. For example,
       * going down 1 from grid position (2,1) is the same as going right 1 from grid position
       * (1,2). To avoid adding states more than once, we keep a list of derivation states we have
       * already added to the candidates heap.
       * 
       * TODO: these should really be keyed on the states themselves instead of a string
       * representation of them.
       */
      derivationTable = new HashSet<DerivationState>();

      /*
       * A Joshua configuration option allows the decoder to output only unique strings. In that
       * case, we keep an list of the frontiers of derivation states extending from this node.
       */
      if (extractUniqueNbest) {
        uniqueStringsTable = new HashSet<String>();
      }

      /*
       * Get the single-best derivation along each of the incoming hyperedges, and add the lot of
       * them to the priority queue of candidates in the form of DerivationState objects.
       * 
       * Note that since the hyperedges are not sorted according to score, the first derivation
       * computed here may not be the best. But since the loop over all hyperedges seeds the entire
       * candidates list with the one-best along each of them, when the candidate heap is polled
       * afterwards, we are guaranteed to have the best one.
       */
      int pos = 0;
      for (HyperEdge edge : node.hyperedges) {
        DerivationState bestState = getBestDerivation(kbestExtractor, node, edge, pos);
        // why duplicate, e.g., 1 2 + 1 0 == 2 1 + 0 1 , but here we should not get duplicate
        if (!derivationTable.contains(bestState)) {
          candHeap.add(bestState);
          derivationTable.add(bestState);
        } else { // sanity check
          throw new RuntimeException(
              "get duplicate derivation in get_candidates, this should not happen"
                  + "\nsignature is " + bestState + "\nl_hyperedge size is "
                  + node.hyperedges.size());
        }
        pos++;
      }

      // TODO: if tem.size is too large, this may cause unnecessary computation, we comment the
      // segment to accommodate the unique nbest extraction
      /*
       * if(tem.size()>global_n){ heap_cands=new PriorityQueue<DerivationState>(new DerivationStateComparator()); for(int i=1;
       * i<=global_n; i++) heap_cands.add(tem.poll()); }else heap_cands=tem;
       */
    }

    // get my best derivation, and recursively add 1best for all my children, used by get_candidates
    // only
    /**
     * This computes the best derivation along a particular hyperedge. It is only called by
     * getCandidates() to initialize the candidates priority queue at each (virtual) node.
     * 
     * @param kbestExtractor
     * @param parentNode
     * @param hyperEdge
     * @param edgePos
     * @return an object representing the best derivation from this node
     */
    private DerivationState getBestDerivation(KBestExtractor kbestExtractor, HGNode parentNode,
        HyperEdge hyperEdge, int edgePos) {
      int[] ranks;
      float cost = 0.0f;

      /*
       * There are two cases: (1) leaf nodes and (2) internal nodes. A leaf node is represented by a
       * hyperedge with no tail nodes.
       */
      if (hyperEdge.getTailNodes() == null) {
        ranks = null;

      } else {
        // "ranks" records which derivation to take at each of the tail nodes. Ranks are 1-indexed.
        ranks = new int[hyperEdge.getTailNodes().size()];

        /* Initialize the one-best at each tail node. */
        for (int i = 0; i < hyperEdge.getTailNodes().size(); i++) { // children is ready
          ranks[i] = 1;
          VirtualNode childVirtualNode = kbestExtractor.getVirtualNode(hyperEdge.getTailNodes()
              .get(i));
          // recurse
          childVirtualNode.lazyKBestExtractOnNode(kbestExtractor, ranks[i]);
        }
      }
      cost = (float) hyperEdge.getBestDerivationScore();

      DerivationState state = new DerivationState(KBestExtractor.this, parentNode, hyperEdge, ranks, cost, edgePos);

      return state;
    }
  };

  public static class DerivationStateComparator implements Comparator<DerivationState> {
    // natural order by cost
    public int compare(DerivationState one, DerivationState another) {
      if (one.getCost() > another.getCost()) {
        return -1;
      } else if (one.getCost() == another.getCost()) {
        return 0;
      } else {
        return 1;
      }
    }
  }

  /**
   * This interface provides a generic way to do things at each stage of a derivation. The
   * DerivationState::visit() function visits every node in a derivation and calls the
   * DerivationVisitor functions both before and after it visits each node. This provides a common
   * way to do different things to the tree (e.g., extract its words, assemble a derivation, and so
   * on) without having to rewrite the node-visiting code.
   * 
   * @author Matt Post <post@cs.jhu.edu>
   */
  public interface DerivationVisitor {
    /**
     * Called before each node's children are visited.
     *
     * @param state the derivation state
     * @param level the tree depth
     * @param tailNodeIndex the tailNodeIndex corresponding to state
     */
    void before(DerivationState state, int level, int tailNodeIndex);

    /**
     * Called after a node's children have been visited.
     * 
     * @param state the derivation state
     * @param level the tree depth
     * @param tailNodeIndex the tailNodeIndex corresponding to state
     */
    void after(DerivationState state, int level, int tailNodeIndex);
  }
  
  /**
   * Assembles a Penn treebank format tree for a given derivation.
   */
  public class TreeExtractor implements DerivationVisitor {

    /* The tree being built. */
    private Tree tree;

    public TreeExtractor() {
      tree = null;
    }

    /**
     * Before visiting the children, find the fragment representation for the current rule,
     * and merge it into the tree we're building.
     */
    @Override
    public void before(DerivationState state, int indent, int tailNodeIndex) {
      HyperEdge edge = state.edge;
      Rule rule = edge.getRule();

      // Skip the special top-level rule
      if (rule == null) {
        return;
      }

      String lhs = Vocabulary.word(rule.getLHS());
      String unbracketedLHS = lhs.substring(1, lhs.length() - 1);

      /* Find the fragment corresponding to this flattened rule in the fragment map; if it's not
       * there, just pretend it's a depth-one rule.
       */
      Tree fragment = Tree.getFragmentFromYield(rule.getEnglishWords());
      if (fragment == null) {
        String subtree = String.format("(%s{%d-%d} %s)", unbracketedLHS, 
            state.parentNode.i, state.parentNode.j, 
            quoteTerminals(rule.getEnglishWords()));
        fragment = Tree.fromString(subtree);
      }
      
      merge(fragment);
    }

    /**
     * Quotes just the terminals in the yield of a tree, represented as a string. This is to force
     * compliance with the Tree class, which interprets all non-quoted strings as nonterminals. 
     * 
     * @param words a string of words representing a rule's yield
     * @return
     */
    private String quoteTerminals(String words) {
      StringBuilder quotedWords = new StringBuilder();
      for (String word: words.split("\\s+"))
        if (word.startsWith("[") && word.endsWith("]"))
          quotedWords.append(String.format("%s ", word));
        else
        quotedWords.append(String.format("\"%s\" ", word));

      return quotedWords.substring(0, quotedWords.length() - 1);
    }

    @Override
    public void after(DerivationState state, int indent, int tailNodeIndex) {
      // do nothing
    }

    public String toString() {
      return tree.unquotedString();
    }

    /**
     * Either set the root of the tree or merge this tree by grafting it onto the first nonterminal
     * in the yield of the parent tree.
     * 
     * @param fragment
     */
    private void merge(Tree fragment) {
      if (tree == null) {
        tree = fragment;
      } else {
        Tree parent = tree.getNonterminalYield().get(0);
        parent.setLabel(Vocabulary.word(fragment.getLabel()));
        parent.setChildren(fragment.getChildren());
      }
    }
  }

  /**
   * Assembles an informative version of the derivation. Each rule is printed as it is encountered.
   * Don't try to parse this output; make something that writes out JSON or something, instead.
   * 
   * @author Matt Post <post@cs.jhu.edu
   */
  public class DerivationExtractor implements DerivationVisitor {

    StringBuffer sb;

    public DerivationExtractor() {
      sb = new StringBuffer();
    }

    @Override
    public void before(DerivationState state, int indent, int tailNodeIndex) {

      HyperEdge edge = state.edge;
      Rule rule = edge.getRule();

      if (rule != null) {

        for (int i = 0; i < indent * 2; i++)
          sb.append(" ");

        final FeatureVectorExtractor extractor = new FeatureVectorExtractor(featureFunctions, sentence);
        extractor.before(state, indent, tailNodeIndex);
        final FeatureVector transitionFeatures = extractor.getFeatures();

        // sb.append(rule).append(" ||| " + features + " ||| " +
        // KBestExtractor.this.weights.innerProduct(features));
        sb.append(String.format("%d-%d", state.parentNode.i, state.parentNode.j));
        sb.append(" ||| " + Vocabulary.word(rule.getLHS()) + " -> "
            + Vocabulary.getWords(rule.getFrench()) + " /// " + rule.getEnglishWords());
        sb.append(" |||");
        for (DPState dpState : state.parentNode.getDPStates()) {
          sb.append(" " + dpState);
        }
        sb.append(" ||| " + transitionFeatures);
        sb.append(" ||| " + weights.innerProduct(transitionFeatures));
        if (rule.getAlignment() != null)
          sb.append(" ||| " + Arrays.toString(rule.getAlignment()));
        sb.append("\n");
      }
    }

    public String toString() {
      return sb.toString();
    }

    @Override
    public void after(DerivationState state, int level, int tailNodeIndex) {}
  }

  @Override
  public Iterator<DerivationState> iterator() {
    return this;
  }

  @Override
  public boolean hasNext() {
    if (this.nextDerivation == null) {
      this.derivationCounter++;
      if (this.derivationCounter <= this.maxDerivations) {
        VirtualNode virtualNode = getVirtualNode(hyperGraph.goalNode);
        this.nextDerivation = virtualNode.lazyKBestExtractOnNode(this, derivationCounter);
      }
      return this.nextDerivation != null;
    }
    
    return true;
  }

  @Override
  public DerivationState next() {
    if (this.hasNext()) {
      DerivationState returnDerivation = this.nextDerivation;
      this.nextDerivation = null;
      return returnDerivation;
    }
    return null;
  }
}
