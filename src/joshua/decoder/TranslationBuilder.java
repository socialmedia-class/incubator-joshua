package joshua.decoder;

import joshua.decoder.ff.FeatureFunction;
import joshua.decoder.ff.FeatureVector;
import joshua.decoder.ff.lm.StateMinimizingLanguageModel;
import joshua.decoder.hypergraph.DerivationState;
import joshua.decoder.hypergraph.KBestExtractor.Side;
import joshua.decoder.io.DeNormalize;
import joshua.decoder.segment_file.Sentence;
import joshua.decoder.segment_file.Token;
import joshua.util.FormatUtils;

import java.util.List;

public class TranslationBuilder {

  private final Sentence sentence;
  private final JoshuaConfiguration config;
  private final DerivationState derivation;
  private final List<FeatureFunction> featureFunctions;
  
  private Translation translation;
  
  public TranslationBuilder(Sentence sentence, DerivationState derivation, 
      List<FeatureFunction> featureFunctions, JoshuaConfiguration config) {
    this.sentence = sentence;
    this.derivation = derivation;
    this.featureFunctions = featureFunctions;
    this.config = config;
    
    if (this.derivation != null) {
      this.translation = new Translation(sentence, derivation.getHypothesis(), derivation.getCost());
    } else {
      this.translation = new Translation(sentence, null, 0.0f);
    }
  }
  
  /**
   * Returns the underlying translation object that was being built. Once this is called, it
   * the TranslationFactory object assumes that the hypergraph is no longer needed.
   * 
   * @return the built Translation object
   */
  public Translation translation() {
    return this.translation;
  }

  public TranslationBuilder formattedTranslation(String format) {

    // TODO: instead of calling replace() a million times, walk through yourself and find the
    // special characters, and then replace them.  If you do this from the right side the index
    // replacement should be a lot more efficient than what we're doing here, particularly since
    // all these arguments get evaluated whether they're used or not

    String output = format
        .replace("%s", translation.toString())
        .replace("%e", derivation.getHypothesis(Side.SOURCE))
        .replace("%S", DeNormalize.processSingleLine(translation.toString()))
        .replace("%c", String.format("%.3f", translation.score()))
        .replace("%i", Integer.toString(sentence.id()));

    if (output.contains("%a")) {
      this.withAlignments().translation();
      output = output.replace("%a", translation.getWordAlignment().toString());
    }

    if (config.outputFormat.contains("%f")) {
      this.withFeatures();
      final FeatureVector features = translation.getFeatures();
      output = output.replace("%f", config.moses ? features.mosesString() : features.toString());
    }
    
    if (output.contains("%t")) {
      // TODO: also store in Translation objection
      output = output.replace("%t", derivation.getTree());
    }

    /* %d causes a derivation with rules one per line to be output */
    if (output.contains("%d")) {
      // TODO: also store in Translation objection
      output = output.replace("%d", derivation.getDerivation());
    }

    translation.setFormattedTranslation(maybeProjectCase(derivation, output));
    return this;
  }

  /** 
   * Stores the features
   * 
   * @return
   */
  public TranslationBuilder withFeatures() {
    translation.setFeatures(derivation.getFeatures());
    return this;
  }
  
  public TranslationBuilder withAlignments() {
    translation.setWordAlignment(derivation.getWordAlignment());
    return this;
  }
  
  /**
   * If requested, projects source-side lettercase to target, and appends the alignment from
   * to the source-side sentence in ||s.
   * 
   * @param hypothesis
   * @param state
   * @return
   */
  private String maybeProjectCase(DerivationState derivation, String hypothesis) {
    String output = hypothesis;

    if (config.project_case) {
      String[] tokens = hypothesis.split("\\s+");
      List<List<Integer>> points = derivation.getWordAlignment().toFinalList();
      for (int i = 0; i < points.size(); i++) {
        List<Integer> target = points.get(i);
        for (int source: target) {
          Token token = sentence.getTokens().get(source + 1); // skip <s>
          String annotation = "";
          if (token != null && token.getAnnotation("lettercase") != null)
            annotation = token.getAnnotation("lettercase");
          if (source != 0 && annotation.equals("upper"))
            tokens[i] = FormatUtils.capitalize(tokens[i]);
          else if (annotation.equals("all-upper"))
            tokens[i] = tokens[i].toUpperCase();
        }
      }

      output = String.join(" ",  tokens);
    }

    return output;
  }
}
