# Research TODO - Differential Vision for Computer-Use Agents

## 1. Differential Vision Encoding

### Concept
Instead of re-encoding full screenshots at each conversational turn in computer-use agents, explore encoding only the changed patches/regions to improve speed.

### Technical Questions to Investigate
- **Feasibility of Feature Caching**: Can we cache the image encoder's output features from previous frames?
- **Differential Encoding**: Is it possible to encode only changed patches and merge them with cached features?
- **Feature Space Compatibility**: Since the image encoder produces new features for each image, how can we ensure compatibility between cached and new differential features?

### Implementation Ideas
- Track pixel-level changes between consecutive screenshots
- Identify changed regions/patches (e.g., using image diff algorithms)
- Encode only changed patches through the vision encoder
- Develop a method to merge new patch features with cached full-frame features
- Benchmark speed improvements vs. accuracy trade-offswhcih

### Challenges
- Vision transformers typically expect fixed-size inputs
- Positional encodings might need special handling for partial updates
- Feature interpolation/merging strategy needs to preserve spatial relationships

### Metrics to Track
- Encoding speed improvement (ms per frame)
- Memory usage for caching
- Accuracy degradation (if any) on downstream tasks
- Bandwidth reduction for distributed systems

---

## 2. Attention Heatmap-Based Click Targeting

### Concept
Use the last layer attention heatmap from the vision model to automatically generate bounding boxes around regions of interest, then use these to guide click actions.

### Technical Approach
- Extract attention weights from the final transformer layer before decoding
- Convert attention maps to spatial heatmaps
- Apply thresholding/clustering to identify high-attention regions
- Generate bounding boxes around these regions
- Use bounding boxes to constrain or guide click action selection

### Implementation Steps
1. **Attention Extraction**
   - Hook into the last attention layer of the vision model
   - Extract attention weights for [CLS] token or action-relevant tokens
   
2. **Heatmap Processing**
   - Reshape attention to spatial dimensions
   - Apply Gaussian smoothing if needed
   - Normalize values to [0, 1] range

3. **Bounding Box Generation**
   - Threshold heatmap (adaptive or fixed threshold)
   - Use connected components or clustering to identify regions
   - Generate minimal bounding boxes around regions
   - Rank boxes by attention strength

4. **Action Integration**
   - When model predicts a click action, use attention-derived bboxes
   - Either constrain clicks to high-attention regions or use as additional input
   - Could provide multiple candidate click locations ranked by attention

### Potential Benefits
- More interpretable click decisions
- Reduced search space for click locations
- Could improve accuracy on small UI elements
- Natural alignment between what model "sees" and where it acts

### Experiments to Run
- Compare click accuracy with/without attention guidance
- Measure if attention correlates with correct click locations
- Test on different UI types (web, desktop, mobile)
- Evaluate on tasks requiring precise targeting vs. general navigation

---

## Next Steps

### Priority 1: Differential Vision
- [ ] Implement basic patch detection algorithm
- [ ] Create differential encoder wrapper
- [ ] Set up benchmarking framework
- [ ] Test on sample screenshot sequences

### Priority 2: Attention-Based Clicking  
- [ ] Extract and visualize attention heatmaps
- [ ] Implement bbox generation from heatmaps
- [ ] Integrate with existing click prediction
- [ ] Evaluate on computer-use benchmarks

### Infrastructure Needs
- [ ] Set up experiment tracking (wandb/mlflow)
- [ ] Create evaluation datasets with screenshot sequences
- [ ] Implement metrics collection pipeline
- [ ] Build visualization tools for debugging

### Research Questions
- How much speedup can differential encoding provide?
- Does attention naturally highlight clickable elements?
- Can we combine both approaches for better performance?
- What's the impact on multi-turn conversation coherence?