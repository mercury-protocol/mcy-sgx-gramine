ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

.PHONY: all
all: sgxapp.manifest
all: sgxapp.manifest.sgx sgxapp.sig sgxapp.token

sgxapp.manifest: sgxapp.manifest.template
	gramine-manifest \
		-Darch_libdir=$(ARCH_LIBDIR) \
		$< > $@

# Make on Ubuntu <= 20.04 doesn't support "Rules with Grouped Targets" (`&:`),
# for details on this workaround see
# https://github.com/gramineproject/gramine/blob/e8735ea06c/CI-Examples/helloworld/Makefile
sgxapp.manifest.sgx sgxapp.sig: sgx_sign
	@:

.INTERMEDIATE: sgx_sign
sgx_sign: sgxapp.manifest
	gramine-sgx-sign \
		--manifest $< \
		--output $<.sgx

sgxapp.token: sgxapp.sig
	gramine-sgx-get-token --output $@ --sig $<

.PHONY: clean
clean:
	$(RM) *.token sgxapp.sig *.manifest.sgx *.manifest

.PHONY: distclean
distclean: clean
	$(RM) *.sig *.quote *.cert *.report attestation_report.json \
		local_secret_key local_public_key data_shared_secret_key model_shared_secret_key \
		data_public_key model_public_key encrypted_data.csv encrypted_model.py encrypted_trained_model.pkl
