.PHONY: all
all: sgxapp.manifest
all: sgxapp.manifest.sgx sgxapp.sig

sgxapp.manifest: sgxapp.manifest.template
	gramine-manifest $< > $@

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
	$(RM) -rf io_files local_secret_key
