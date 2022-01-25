# Transport

Common implementation details for gRPC Transports.

Transports multiplex messages across some single connection. In ext/ there are
implementations atop [a custom http2 implementation](//core/ext/transport/chttp2/README.md)
and atop [cronet](//core/ext/transport/cronet/README.md).
