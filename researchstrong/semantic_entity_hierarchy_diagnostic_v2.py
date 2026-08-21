from pathlib import Path
import builtins
import numpy as np
import semantic_entity_hierarchy_diagnostic as h

# Compatibility-only fix: semantic_entity_hierarchy_diagnostic used bool(np.ndarray)
# once when checking whether an argsort result was non-empty. Preserve every
# scientific choice and replace ndarray truth-testing with len(x) > 0.
h.bool = lambda x: (len(x) > 0) if isinstance(x, np.ndarray) else builtins.bool(x)
h.OUT = Path(__file__).with_name('semantic_entity_hierarchy_diagnostic_result.json')

if __name__ == '__main__':
    h.main()
