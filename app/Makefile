ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)
PILLOW_PATH ?= $(shell pip3 show pillow | grep Location: | cut -d" " -f2)/

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
else
GRAMINE_LOG_LEVEL = error
endif

.PHONY: all
all: ra.manifest
all: ra.manifest.sgx ra.sig ra.token

ra.manifest: ra.manifest.template
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Dpillow_path=$(PILLOW_PATH) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dentrypoint=$(realpath $(shell sh -c "command -v python3")) \
		$< > $@

# Make on Ubuntu <= 20.04 doesn't support "Rules with Grouped Targets" (`&:`),
# for details on this workaround see
# https://github.com/gramineproject/gramine/blob/e8735ea06c/CI-Examples/helloworld/Makefile
ra.manifest.sgx ra.sig: sgx_sign
	@:

.INTERMEDIATE: sgx_sign
sgx_sign: ra.manifest
	gramine-sgx-sign \
		--manifest $< \
		--output $<.sgx

ra.token: ra.sig
	gramine-sgx-get-token --output $@ --sig $<

.PHONY: clean
clean:
	$(RM) *.token ra.sig *.manifest.sgx *.manifest

.PHONY: distclean
distclean: clean
	$(RM) *.sig *.quote *.cert *.report local_secret shared_secret_model shared_secret_data
