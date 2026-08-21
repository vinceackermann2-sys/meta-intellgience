from pathlib import Path
import builtins
import numpy as np
import semantic_entity_hierarchy_diagnostic as h

h.bool = lambda x: (len(x) > 0) if isinstance(x, np.ndarray) else builtins.bool(x)
h.N = 30
h.OUT = Path(__file__).with_name('semantic_entity_hierarchy_smoke_result.json')

if __name__ == '__main__':
    h.main()
