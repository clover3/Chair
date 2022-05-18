import six
from tensorflow.python.training.tracking.util import _ObjectGraphProtoPrettyPrinter, CheckpointLoadStatus


def debug_assert_consumed(cpls: CheckpointLoadStatus):
    self = cpls
    pretty_printer = _ObjectGraphProtoPrettyPrinter(
        self._checkpoint.object_graph_proto)
    # self.assert_existing_objects_matched()
    print("Graph has {} nodes".format(len(self._checkpoint.object_graph_proto.nodes)))
    for node_id, node in enumerate(self._checkpoint.object_graph_proto.nodes):
        if not node.attributes:
            print("Ignore {}".format(pretty_printer.node_names[node_id]))
            # Only raise exceptions for the nodes with attributes themselves. Either
            # they're ultimately not important, or they have a child with an
            # attribute.
            continue

        print("Count {}".format(pretty_printer.node_names[node_id]))
        trackable = self._checkpoint.object_by_proto_id.get(node_id, None)
        if trackable is None:
            print("Trackable is None")
            raise AssertionError("Unresolved object in checkpoint {}: {}"
                                 .format(pretty_printer.node_names[node_id], node))
    if self._checkpoint.slot_restorations:
        # Sanity check; this collection should be clear if everything has been
        # restored.
        raise AssertionError("Unresolved slot restorations: %s" %
                             (self._checkpoint.slot_restorations,))
    if self._checkpoint.unused_attributes:
        unused_attribute_messages = []
        for node_id, attribute in six.iteritems(
                self._checkpoint.unused_attributes):
            obj = self._checkpoint.object_by_proto_id[node_id]
            unused_attribute_messages.append(
                "{} ({}): {}"
                    .format(pretty_printer.node_names[node_id], obj, attribute))
        raise AssertionError(
            ("Unused attributes in these objects (the attributes exist in the "
             "checkpoint but were not restored):\n{}")
                .format("\n".join(unused_attribute_messages)))