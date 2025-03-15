#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from pomegranate import HiddenMarkovModel, State, DiscreteDistribution

# Define observable notes
notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
note_to_pitch = {'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71}  # MIDI note numbers

# Ensure probabilities sum to 1
original_calm = DiscreteDistribution({'C': 0.3, 'D': 0.2, 'E': 0.1, 'F': 0.1, 'G': 0.3})
original_energetic = DiscreteDistribution({'C': 0.1, 'D': 0.1, 'E': 0.3, 'F': 0.3, 'G': 0.1, 'A': 0.1})
original_melancholic = DiscreteDistribution({'D': 0.3, 'E': 0.1, 'F': 0.3, 'G': 0.1, 'B': 0.2})

modified_calm = DiscreteDistribution({'C': 0.4, 'D': 0.1, 'E': 0.1, 'F': 0.1, 'G': 0.3})
modified_energetic = DiscreteDistribution({'C': 0.05, 'D': 0.1, 'E': 0.25, 'F': 0.25, 'G': 0.05, 'A': 0.3})
modified_melancholic = DiscreteDistribution({'D': 0.2, 'E': 0.1, 'F': 0.4, 'G': 0.1, 'B': 0.2})

# Function to run and plot HMM

def run_hmm_and_plot(calm_dist, energetic_dist, melancholic_dist, title):
    calm = State(calm_dist, name='Calm')
    energetic = State(energetic_dist, name='Energetic')
    melancholic = State(melancholic_dist, name='Melancholic')

    hmm = HiddenMarkovModel("Music HMM")
    hmm.add_states(calm, energetic, melancholic)

    # Transitions
    hmm.add_transition(hmm.start, calm, 0.5)
    hmm.add_transition(hmm.start, energetic, 0.3)
    hmm.add_transition(hmm.start, melancholic, 0.2)

    hmm.add_transition(calm, calm, 0.6)
    hmm.add_transition(calm, energetic, 0.2)
    hmm.add_transition(calm, melancholic, 0.2)

    hmm.add_transition(energetic, energetic, 0.5)
    hmm.add_transition(energetic, calm, 0.3)
    hmm.add_transition(energetic, melancholic, 0.2)

    hmm.add_transition(melancholic, melancholic, 0.6)
    hmm.add_transition(melancholic, energetic, 0.2)
    hmm.add_transition(melancholic, calm, 0.2)

    hmm.add_transition(calm, hmm.end, 0.1)
    hmm.add_transition(energetic, hmm.end, 0.1)
    hmm.add_transition(melancholic, hmm.end, 0.1)

    hmm.bake()

    # Generate melody
    sequence = hmm.sample(length=10)
    print(f"{title} Generated melody:", sequence)
    
    # Plot melody
    plt.figure(figsize=(10, 4))
    note_pitches = [note_to_pitch[note] - 60 for note in sequence]
    plt.plot(range(len(sequence)), note_pitches, marker='o', linestyle='-', color='b')
    plt.yticks(range(len(notes)), notes)
    plt.xlabel("Time Step")
    plt.ylabel("Note")
    plt.title(title)
    plt.grid(True)
    plt.show()

# Run experiments
run_hmm_and_plot(original_calm, original_energetic, original_melancholic, "Original Probability Distributions")
run_hmm_and_plot(modified_calm, modified_energetic, modified_melancholic, "Modified Probability Distributions")


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from pomegranate import HiddenMarkovModel, State, DiscreteDistribution

# Define observable notes
notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
note_to_pitch = {'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71}  # MIDI note numbers

# Define moods
moods = ['Calm', 'Energetic', 'Melancholic']
mood_to_index = {'Calm': 0, 'Energetic': 1, 'Melancholic': 2}

# Define probability distributions for each mood
calm_emissions = DiscreteDistribution({'C': 0.3, 'D': 0.2, 'E': 0.1, 'F': 0.1, 'G': 0.3})
energetic_emissions = DiscreteDistribution({'C': 0.1, 'D': 0.1, 'E': 0.3, 'F': 0.3, 'G': 0.1, 'A': 0.1})
melancholic_emissions = DiscreteDistribution({'D': 0.3, 'E': 0.1, 'F': 0.3, 'G': 0.1, 'B': 0.2})

# Define states
calm = State(calm_emissions, name='Calm')
energetic = State(energetic_emissions, name='Energetic')
melancholic = State(melancholic_emissions, name='Melancholic')

# Create HMM model
hmm = HiddenMarkovModel("Music HMM")
hmm.add_states(calm, energetic, melancholic)

# Define transition probabilities
hmm.add_transition(hmm.start, calm, 0.5)
hmm.add_transition(hmm.start, energetic, 0.3)
hmm.add_transition(hmm.start, melancholic, 0.2)

hmm.add_transition(calm, calm, 0.6)
hmm.add_transition(calm, energetic, 0.2)
hmm.add_transition(calm, melancholic, 0.2)

hmm.add_transition(energetic, energetic, 0.5)
hmm.add_transition(energetic, calm, 0.3)
hmm.add_transition(energetic, melancholic, 0.2)

hmm.add_transition(melancholic, melancholic, 0.6)
hmm.add_transition(melancholic, energetic, 0.2)
hmm.add_transition(melancholic, calm, 0.2)

hmm.add_transition(calm, hmm.end, 0.1)
hmm.add_transition(energetic, hmm.end, 0.1)
hmm.add_transition(melancholic, hmm.end, 0.1)

# Finalize the model
hmm.bake()

# Generate melody with mood transitions
sequence, state_sequence = hmm.sample(length=10, path=True)

# Extract state (mood) names from state_sequence
mood_sequence = [state.name for state in state_sequence[1:-1]]  # Exclude start/end states

print("Generated melody:", sequence)
print("Mood transitions:", mood_sequence)

# Convert mood sequence to indices for plotting
mood_indices = [mood_to_index[mood] for mood in mood_sequence]

# Plot the mood transitions
plt.figure(figsize=(10, 4))
plt.plot(range(len(mood_indices)), mood_indices, marker='o', linestyle='-', color='r', label="Mood")
plt.yticks(range(len(moods)), moods)
plt.xlabel("Time Step")
plt.ylabel("Mood")
plt.title("Mood Transitions Over Time")
plt.grid(True)
plt.legend()
plt.show()

# Plot the melody
plt.figure(figsize=(10, 4))
note_pitches = [note_to_pitch[note] - 60 for note in sequence]
plt.plot(range(len(sequence)), note_pitches, marker='o', linestyle='-', color='b', label="Melody")
plt.yticks(range(len(notes)), notes)
plt.xlabel("Time Step")
plt.ylabel("Note")
plt.title("Generated Melody on a Music Staff")
plt.grid(True)
plt.legend()
plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from pomegranate import HiddenMarkovModel, State, DiscreteDistribution

# Define observable notes
notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
note_to_pitch = {'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71}  # MIDI note numbers

# Define moods
moods = ['Calm', 'Energetic', 'Melancholic']
mood_to_index = {'Calm': 0, 'Energetic': 1, 'Melancholic': 2}

# Define probability distributions for each mood
calm_emissions = DiscreteDistribution({'C': 0.3, 'D': 0.2, 'E': 0.1, 'F': 0.1, 'G': 0.3})
energetic_emissions = DiscreteDistribution({'C': 0.1, 'D': 0.1, 'E': 0.3, 'F': 0.3, 'G': 0.1, 'A': 0.1})
melancholic_emissions = DiscreteDistribution({'D': 0.3, 'E': 0.1, 'F': 0.3, 'G': 0.1, 'B': 0.2})

# Define states
calm = State(calm_emissions, name='Calm')
energetic = State(energetic_emissions, name='Energetic')
melancholic = State(melancholic_emissions, name='Melancholic')

# Create HMM model
hmm = HiddenMarkovModel("Music HMM")
hmm.add_states(calm, energetic, melancholic)

# Define transition probabilities
hmm.add_transition(hmm.start, calm, 0.5)
hmm.add_transition(hmm.start, energetic, 0.3)
hmm.add_transition(hmm.start, melancholic, 0.2)

hmm.add_transition(calm, calm, 0.6)
hmm.add_transition(calm, energetic, 0.2)
hmm.add_transition(calm, melancholic, 0.2)

hmm.add_transition(energetic, energetic, 0.5)
hmm.add_transition(energetic, calm, 0.3)
hmm.add_transition(energetic, melancholic, 0.2)

# Adjusted Melancholic transition probabilities
hmm.add_transition(melancholic, melancholic, 0.4)  # Reduced from 0.6 to 0.4
hmm.add_transition(melancholic, energetic, 0.3)  # Increased to balance
hmm.add_transition(melancholic, calm, 0.3)  # Increased to balance

hmm.add_transition(calm, hmm.end, 0.1)
hmm.add_transition(energetic, hmm.end, 0.1)
hmm.add_transition(melancholic, hmm.end, 0.1)

# Finalize the model
hmm.bake()

# Generate melody with mood transitions
sequence, state_sequence = hmm.sample(length=10, path=True)

# Extract state (mood) names from state_sequence
mood_sequence = [state.name for state in state_sequence[1:-1]]  # Exclude start/end states

print("Generated melody:", sequence)
print("Mood transitions:", mood_sequence)

# Convert mood sequence to indices for plotting
mood_indices = [mood_to_index[mood] for mood in mood_sequence]

# Plot the mood transitions
plt.figure(figsize=(10, 4))
plt.plot(range(len(mood_indices)), mood_indices, marker='o', linestyle='-', color='r', label="Mood")
plt.yticks(range(len(moods)), moods)
plt.xlabel("Time Step")
plt.ylabel("Mood")
plt.title("Mood Transitions Over Time")
plt.grid(True)
plt.legend()
plt.show()

# Plot the melody
plt.figure(figsize=(10, 4))
note_pitches = [note_to_pitch[note] - 60 for note in sequence]
plt.plot(range(len(sequence)), note_pitches, marker='o', linestyle='-', color='b', label="Melody")
plt.yticks(range(len(notes)), notes)
plt.xlabel("Time Step")
plt.ylabel("Note")
plt.title("Generated Melody on a Music Staff")
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:




