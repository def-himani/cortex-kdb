---
type: Interface
title: Joint
iri: https://kdb.aegean.ai/core/interfaces/joint
properties:
- property: https://kdb.aegean.ai/core/properties/title
  required: true
links:
- name: parent
  target: https://kdb.aegean.ai/core/interfaces/link
  cardinality: many-to-one
  required: true
- name: child
  target: https://kdb.aegean.ai/core/interfaces/link
  cardinality: many-to-one
  required: true
generated:
  by: okf-seed/1
  at: '2026-11-25T21:02:19Z'
---

A connection between two links that allows one kind of relative motion. It joins a parent link to a child link.
