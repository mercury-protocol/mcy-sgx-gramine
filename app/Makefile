ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)
PILLOW_PATH ?= $(shell pip3 show pillow | grep Location: | cut -d" " -f2)/

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
else
GRAMINE_LOG_LEVEL = error
endif

.PHONY: all
all: sgxapp.manifest
all: sgxapp.manifest.sgx sgxapp.sig sgxapp.token

sgxapp.manifest: sgxapp.manifest.template
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Dpillow_path=$(PILLOW_PATH) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dentrypoint=$(realpath $(shell sh -c "command -v python3")) \
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
	$(RM) *.sig *.quote *.cert *.report local_secret_key model_shared_secret_key data_shared_secret_key
