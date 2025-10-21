# Attention clicking


# Input image + action(from benchmark, reasoning) + prompt(pyautoguicommands)
# Predict the ACTION like pyautogui.click or  pyautogui.moveTo

# Take attention map from last layer of the vision model 
# Make "heatmap" from the attention, focused on the ACTION
# Make bounding boxes from the heatmap using simple CV
# Feed the bounding boxes back into the prompt after the action and ask it to pick one, multiple choice style
