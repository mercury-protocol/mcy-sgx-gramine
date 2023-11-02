ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)
ENTRYPOINT ?= $(realpath $(shell sh -c "command -v python3"))

.PHONY: all
all: sgxapp.manifest
all: sgxapp.manifest.sgx sgxapp.sig

RA_CLIENT_SPID ?=

sgxapp.manifest: sgxapp.manifest.template
	gramine-manifest \
	-DARCH_LIBDIR=$(ARCH_LIBDIR) \
	-DENTRYPOINT=$(ENTRYPOINT) \
	-DRA_CLIENT_SPID=$(RA_CLIENT_SPID) \
	$< > $@

sgxapp.manifest.sgx sgxapp.sig: sgx_sign
	@:

.INTERMEDIATE: sgx_sign
sgx_sign: sgxapp.manifest
	gramine-sgx-sign --manifest $< --output $<.sgx

.PHONY: clean
clean:
	$(RM) *.token sgxapp.sig *.manifest.sgx *.manifest

.PHONY: distclean
distclean: clean
	$(RM) -rf input output local_secret_key
