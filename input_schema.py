INPUT_SCHEMA = {
  "inputs": [
    {
      "data": [
        "I have a problem with my iphone that needs to be resolved asap!!"
      ],
      "name": "text",
      "shape": [
        1
      ],
      "datatype": "BYTES"
    },
    {
      "data": [
        "urgent",
        "not urgent",
        "phone",
        "tablet",
        "computer"
      ],
      "name": "candidate_labels",
      "shape": [
        5
      ],
      "datatype": "BYTES"
    }
  ]
}