# State Class Specification

## State Class Name
`State`

## State Fields

### topic
- **Data Type**: `str`
- **Annotation**: None
- **Description**: The topic provided when the graph is invoked, used as input for joke generation

### joke
- **Data Type**: `str`
- **Annotation**: None
- **Description**: The generated joke based on the provided topic, created by the llm_call_generator node and shown to the human

### funny
- **Data Type**: `bool`
- **Annotation**: None
- **Description**: The evaluation result indicating whether the joke is considered funny or not, determined by the llm_call_evaluator node (randomly judges as funny 50% of the time)

## Usage Notes

- The `topic` field is provided as input when the graph is invoked
- The `joke` field stores the generated joke content for display to the human
- The `funny` field stores the evaluation result used by the `route_joke` routing function to determine whether to regenerate the joke or end the process