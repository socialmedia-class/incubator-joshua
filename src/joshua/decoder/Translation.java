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
package joshua.decoder;

import static joshua.util.FormatUtils.removeSentenceMarkers;
import static joshua.util.FormatUtils.unescapeSpecialSymbols;

import java.util.List;

import joshua.decoder.ff.FeatureVector;
import joshua.decoder.hypergraph.WordAlignmentState;
import joshua.decoder.segment_file.Sentence;


/**
 * This class represents translated input objects (sentences or lattices). It is aware of the source
 * sentence and id and contains the decoded hypergraph. Translation objects are returned by
 * DecoderThread instances to the InputHandler, where they are assembled in order for output.
 * 
 * @author Matt Post <post@cs.jhu.edu>
 */

public class Translation {
  private final Sentence sourceSentence;
  private final String rawTranslation;
  private final String translation;
  private final float translationScore;
  private String formattedTranslation;
  private List<List<Integer>> translationWordAlignments;
  private float extractionTime;
  private FeatureVector features;
  private WordAlignmentState wordAlignment;

  public Translation(final Sentence source, final String output, final float cost) {
    this.sourceSentence = source;
    this.rawTranslation = output;
    this.translationScore = cost;
    
    this.translation = unescapeSpecialSymbols(removeSentenceMarkers(rawTranslation));
    
//    final long startTime = System.currentTimeMillis();
//    this.extractionTime = (System.currentTimeMillis() - startTime) / 1000.0f;
  }

  public Sentence getSourceSentence() {
    return this.sourceSentence;
  }

  public float score() {
    return translationScore;
  }

  /**
   * Returns a list of target to source alignments.
   */
  public List<List<Integer>> getTranslationWordAlignments() {
    return translationWordAlignments;
  }
  
  /**
   * Time taken to build output information from the hypergraph.
   */
  public Float getExtractionTime() {
    return extractionTime;
  }
  
  public int id() {
    return sourceSentence.id();
  }

  @Override
  public String toString() {
    return this.translation;
  }
  
  public String rawTranslation() {
    return this.rawTranslation;
  }

  public void setFormattedTranslation(String formattedTranslation) {
    this.formattedTranslation = formattedTranslation;
  }
  
  public String getFormattedTranslation() {
    return this.formattedTranslation;
  }

  public void setFeatures(FeatureVector features) {
    this.features = features;
  }
  
  public FeatureVector getFeatures() {
    return this.features;
  }

  public void setWordAlignment(WordAlignmentState wordAlignment) {
    this.wordAlignment = wordAlignment;
  }

  public Object getWordAlignment() {
    return this.wordAlignment;
  }
}
