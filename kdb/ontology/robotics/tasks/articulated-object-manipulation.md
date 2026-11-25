---
type: Term
title: Articulated object manipulation
pref_label: Articulated object manipulation
alt_labels:
- opening a drawer
- closing a drawer
- open drawer
- close drawer
- drawer opening
iri: https://kdb.aegean.ai/robotics/tasks/articulated-object-manipulation
broader:
- /ontology/robotics/tasks/manipulation.md
related_match:
- http://www.ease-crc.org/ont/SOMA.owl#Opening
- http://www.ease-crc.org/ont/SOMA.owl#Closing
generated:
  by: okf-seed/1
  at: '2026-11-25T21:38:58Z'
sources:
- id: seed-articulated-object-manipulation-1
  resource: /papers/2112.03227.pdf
  page: 3
  quote: The environments contain a 7-DOF Franka Emika Panda robot arm with a parallel
    gripper and a desk with a sliding door and a drawer that can be opened and closed.
  bbox:
  - 47.82
  - 279.86
  - 300.02
  - 325.68
- id: seed-articulated-object-manipulation-2
  resource: /papers/2308.12952.pdf
  page: 3
  quote: We also count motions that are the reverse of each other (e.g. opening and
    closing a drawer) as a single skill.
  bbox:
  - 111.49
  - 80.03
  - 504.0
  - 98.95
---

Opening and closing the drawer of the CALVIN desk is an example of articulated object manipulation.[^seed-articulated-object-manipulation-1] Opening and closing a drawer count as one skill because one is the reverse of the other.[^seed-articulated-object-manipulation-2]

[^seed-articulated-object-manipulation-1]: arXiv 2112.03227, page 3
[^seed-articulated-object-manipulation-2]: arXiv 2308.12952, page 3
