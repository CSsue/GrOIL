#!/bin/bash

SOURCE_FILES=(
#   "contract_1_term_life.graphml"
#   "contract_2_whole_life.graphml"
  "contract_3_universal_life.graphml"
  "contract_4_variable_universal.graphml"
  "contract_5_final_expense.graphml"
  "contract_6_group_term.graphml"
  "contract_7_term_with_riders.graphml"
  "contract_8_joint_survivorship.graphml"
  "contract_9_return_of_premium.graphml"
#   "contract_10_indexed_universal.graphml"
)

for SOURCE_FILE in "${SOURCE_FILES[@]}"; do
  if [[ "$SOURCE_FILE" =~ contract_([0-9]+)_ ]]; then
    CONTRACT_ID="C${BASH_REMATCH[1]}"
  else
    echo "Skipping ${SOURCE_FILE}: could not extract contract ID"
    continue
  fi

  echo "Running ABox population for ${SOURCE_FILE} with contract ID ${CONTRACT_ID}"

  python src/udh_to_ontology/ontology_generation/abox_population.py \
    --source-file "${SOURCE_FILE}" \
    --contract-id "${CONTRACT_ID}"
done