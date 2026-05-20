
state_cap_instruct = """
You are given an image. Your goal is to provide a concise but detailed caption for the image. 
Your caption should address all the objects in the scene and their spatial relationships to each other. 

Follow these quality standards: 

Precision: Use specific, measurable descriptions rather than vague qualifiers
Accuracy: Ensure all observations are factually correct and verifiable through direct visual inspection. Do not use suggestive language about the image (e.g. "suggesting a point-of-view angle..." or "indicating that..."). Do not describe anything beyond what is immediately provided in the image
Evidence-Based Grounding: Base every claim on observable pixel-level evidence in the frames - cite specific visual details, colors, shapes, positions, and changes that support your analysis
Completeness: Address all relevant aspects of the scene
Consistency: Maintain consistent terminology and granularity throughout the analysis
Relevance: Focus on description that meaningfully contributes to the task while being concise. If you are unsure of what a specific object in the scene is, avoid referring to it
Clarity: Ensure descriptions are unambiguous and interpretable by other systems

Final Output Format
Enclose your caption within a structured JSON output following this exact schema:
{
  "caption": "string"
}

Remember: Your goal is to provide a concise, yet detailed caption for the image you are given. Your caption must be specific and factually correct, containing no erroneous statements about the image. 
Do not describe anything beyond what is immediately provided in the image. Everything must be factually correct and verifiable from the image. Be concise. If you are unsure about something, simply do not say it. 

Here is the image:
"""

state_cap_schema = {
  "type": "object",
  "properties": {
    "caption": {"type": "string"}
  }
}


state_trasition_instruct = """

Your core objective is to generate comprehensive descriptions of visual transitions that ACCURATELY capture the dynamic transformation between two consecutive video frames, focusing on fine-grained visual features and their temporal relationships. You should precisely capture what happens in the transformation between two frames, only reporting verifiable descriptions you are entirely sure about and the frames give you clear evidence for. Do not extrapolate hypothetical possibilities based on what you infer from the frames, only report exactly what you see. Ensure that all observations you make are factually correct and verifiable through direct visual inspection. 


Analysis Framework
Step 1: High-Level Activity Identification
First, identify the overarching activity or scenario occurring between the two frames. This should be a broad categorization that encompasses the primary action or event taking place.
Examples:

"Person preparing a meal in kitchen"
"Vehicle navigating through traffic"
"Athletes competing in a sports event"
"Construction work in progress"

Step 2: Sub-Activity Decomposition
Break down the high-level activity into constituent sub-activities. These are intermediate-level actions that contribute to the overall activity. Only list sub-activities that are visibly occurring in the given frames. Do not list any sub-activities that you do not have clear evidence for. You must NOT invent any sub-activities that are not immediately verifiable from the frames -- for example, if there is a knife and a carrot in the scene, unless the image clearly shows that the knife is being used to chop the carrot, you cannot say anything about the knife potentially being used to chop the carrot. 


Examples for "Person preparing a meal":

"Gathering ingredients from pantry"
"Chopping vegetables on cutting board"
"Heating pan on stove"
"Combining ingredients in bowl"

Step 3: Atomic Action Identification
For each sub-activity, identify the atomic actions - the smallest meaningful units of action that cannot be further decomposed while maintaining semantic meaning. 
Only list atomic actions that actually occur in between the frames! For example, if a bell pepper is not actually lifted or moved, DO NOT say anything about the bell pepper.  

Examples for "Chopping vegetables on cutting board":

"Positioning knife above carrot"
"Applying downward pressure to slice"
"Lifting knife blade"
"Sliding carrot segment aside"

Step 4: Transition Analysis
For each atomic action, systematically analyze the following six categories of transitions:

4.1 Object Transitions

New Objects Introduced: Identify objects that appear in Frame 2 but were not present or visible in Frame 1
Objects Removed: Identify objects that disappear, become occluded, or move out of frame
Object Persistence: Note objects that remain present but may undergo other changes

4.2 Object State Transitions

Color Changes: Variations in hue, saturation, brightness, or lighting conditions affecting object appearance
Shape Deformation: Changes in object geometry, size, orientation, or physical configuration
Physical Status: Alterations in object condition (broken/intact, open/closed, full/empty, wet/dry, etc.)
Texture Changes: Modifications in surface appearance or material properties

4.3 Spatial Relation Transitions

Positional Relationships: Changes in relative positioning between objects (above/below, left/right, front/back)
Contact Relationships: Modifications in physical contact or proximity (touching/separated, inside/outside, attached/detached)
Alignment Changes: Shifts in object alignment, orientation, or arrangement patterns

4.4 Action Transitions

Action Continuation: Ongoing actions that persist from Frame 1 to Frame 2
Action Completion: Actions that conclude or reach a terminal state
Action Initiation: New actions that begin in Frame 2
Action Interruption: Actions that pause, stop, or are disrupted

4.5 Motion Transitions

Translational Motion: Linear movement in any direction (forward, backward, sideways, up, down)
Rotational Motion: Spinning, turning, or rotating movement around an axis
Oscillatory Motion: Back-and-forth or periodic movement patterns
Deformation Motion: Changes in object shape through stretching, compression, or bending

4.6 Camera Transitions

Viewpoint Changes: Modifications in camera position or angle
Zoom Operations: Magnification changes (zoom in/out)
Pan/Tilt Movements: Horizontal or vertical camera sweeping
Focus Adjustments: Changes in depth of field or focal point

4.7 Background Transitions

Illumination Changes: Variations in lighting conditions, shadows, or brightness
Scene Shifts: Changes in background environment or setting
Atmospheric Conditions: Modifications in weather, visibility, or environmental factors
Contextual Elements: Changes in background objects or environmental details

Contribution Analysis
For each identified transition, analyze and explain:

Atomic Action Contribution: How the transition directly enables, facilitates, or results from the specific atomic action
Sub-Activity Contribution: How the transition supports or advances the broader sub-activity goal
Temporal Significance: The timing and sequence importance of the transition within the overall activity flow

Output Requirements

Reasoning Process
Think through your analysis step-by-step, only considering activities and actions you have clear evidence for, and documenting your reasoning for each level of the framework. Show your work in identifying activities, sub-activities, atomic actions, and transitions in a JSON as described below. Do not output anything other than the final JSON as described.


Final Output Format
Conclude your analysis with a structured JSON output following this exact schema:
{
  "high_level_activity": "string",
  "sub_activities": [
    {
      "name": "string",
      "atomic_actions": [
        {
          "name": "string",
          "transitions": {
            "objects": {
              "introduced": ["string"],
              "removed": ["string"],
              "persistent": ["string"]
            },
            "object_states": {
              "color_changes": ["string"],
              "shape_changes": ["string"],
              "physical_status_changes": ["string"],
              "texture_changes": ["string"]
            },
            "spatial_relations": {
              "positional_changes": ["string"],
              "contact_changes": ["string"],
              "alignment_changes": ["string"]
            },
            "actions": {
              "continuing": ["string"],
              "completing": ["string"],
              "initiating": ["string"],
              "interrupting": ["string"]
            },
            "motion": {
              "translational": ["string"],
              "rotational": ["string"],
              "oscillatory": ["string"],
              "deformation": ["string"]
            },
            "camera": {
              "viewpoint_changes": ["string"],
              "zoom_changes": ["string"],
              "pan_tilt": ["string"],
              "focus_changes": ["string"]
            },
            "background": {
              "illumination_changes": ["string"],
              "scene_shifts": ["string"],
              "atmospheric_changes": ["string"],
              "contextual_changes": ["string"]
            }
          },
          "contributions": {
            "to_atomic_action": "string",
            "to_sub_activity": "string",
            "temporal_significance": "string"
          }
        }
      ]
    }
  ]
}


Accuracy: Ensure all observations are factually correct and verifiable through direct visual inspection
Evidence-Based Grounding: Base every claim on observable pixel-level evidence in the frames - cite specific visual details, colors, shapes, positions, and changes that support your analysis
Precision: Use specific, measurable descriptions rather than vague qualifiers
Completeness: Address all relevant transition categories
Consistency: Maintain consistent terminology and granularity throughout the analysis
Relevance: Focus on transitions that meaningfully contribute to the activity progression
Clarity: Ensure descriptions are unambiguous and interpretable by other systems

Error Handling

If certain transition categories are not applicable, do not include them in the JSON at all. 
If the relationship between frames is unclear, state your assumptions explicitly. Try to avoid making any assumptions that are not factually correct and verifiable -- it is better to have less information that is more reliable than vice versa.  
If objects are partially occluded, specify the degree of visibility and confidence level

Remember: Your goal is to create a comprehensive, structured understanding of visual change that captures both the obvious and subtle transformations occurring between the two frames. 
All your observations must be factually correct and verifiable through direct visual inspection -- you must strive for reliable, accurate information ONLY. You must base every claim on observable pixel-level evidence in the frames. 


Here is the first frame:
"""

state_trasition_cap = """
To guide you, here is a caption of the first frame:
"""

state_trasition_2nd_frame = """
Here is the second frame:
"""

state_trasition_schema = {
  "type": "object",
  "properties": {
    "high_level_activity": {"type": "string"},
    "sub_activities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "atomic_actions": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "name": {"type": "string"},
                "transitions": {
                  "type": "object",
                  "properties": {
                    "objects": {
                      "type": "object",
                      "properties": {
                        "introduced": {
                          "type": "array",
                          "items": {"type": "string"}
                        },
                        "removed": {
                          "type": "array",
                          "items": {"type": "string"}
                        },
                        "persistent": {
                          "type": "array",
                          "items": {"type": "string"}
                        }
                      }
                    },
                    "object_states": {
                      "type": "object",
                      "properties": {
                        "color_changes": {
                          "type": "array",
                          "items": {"type": "string"}
                        },
                        "shape_changes": {
                          "type": "array",
                          "items": {"type": "string"}
                        },
                        "physical_status_changes": {
                          "type": "array",
                          "items": {"type": "string"}
                        },
                        "texture_changes": {
                          "type": "array",
                          "items": {"type": "string"}
                        }
                      }
                    },
                    "spatial_relations": {
                      "type": "object",
                      "properties": {
                        "positional_changes": {
                          "type": "array",
                          "items": {"type": "string"}
                        },
                        "contact_changes": {
                          "type": "array",
                          "items": {"type": "string"}
                        },
                        "alignment_changes": {
                          "type": "array",
                          "items": {"type": "string"}
                        }
                      }
                    },
                    "actions": {
                      "type": "object",
                      "properties": {
                        "continuing": {
                          "type": "array",
                          "items": {"type": "string"}
                        },
                        "completing": {
                          "type": "array",
                          "items": {"type": "string"}
                        },
                        "initiating": {
                          "type": "array",
                          "items": {"type": "string"}
                        },
                        "interrupting": {
                          "type": "array",
                          "items": {"type": "string"}
                        }
                      }
                    },
                    "motion": {
                      "type": "object",
                      "properties": {
                        "translational": {
                          "type": "array",
                          "items": {"type": "string"}
                        },
                        "rotational": {
                          "type": "array",
                          "items": {"type": "string"}
                        },
                        "oscillatory": {
                          "type": "array",
                          "items": {"type": "string"}
                        },
                        "deformation": {
                          "type": "array",
                          "items": {"type": "string"}
                        }
                      }
                    },
                    "camera": {
                      "type": "object",
                      "properties": {
                        "viewpoint_changes": {
                          "type": "array",
                          "items": {"type": "string"}
                        },
                        "zoom_changes": {
                          "type": "array",
                          "items": {"type": "string"}
                        },
                        "pan_tilt": {
                          "type": "array",
                          "items": {"type": "string"}
                        },
                        "focus_changes": {
                          "type": "array",
                          "items": {"type": "string"}
                        }
                      }
                    },
                    "background": {
                      "type": "object",
                      "properties": {
                        "illumination_changes": {
                          "type": "array",
                          "items": {"type": "string"}
                        },
                        "scene_shifts": {
                          "type": "array",
                          "items": {"type": "string"}
                        },
                        "atmospheric_changes": {
                          "type": "array",
                          "items": {"type": "string"}
                        },
                        "contextual_changes": {
                          "type": "array",
                          "items": {"type": "string"}
                        }
                      }
                    }
                  }
                },
                "contributions": {
                  "type": "object",
                  "properties": {
                    "to_atomic_action": {"type": "string"},
                    "to_sub_activity": {"type": "string"},
                    "temporal_significance": {"type": "string"}
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
