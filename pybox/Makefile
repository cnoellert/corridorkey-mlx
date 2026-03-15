PYBOX_DIR  = /opt/corridorkey/pybox
REF_DIR    = /opt/corridorkey/reference

PYBOX_FILES = corridorkey_pybox.py \
              corridorkey_daemon_mlx.py \
              corridorkey_daemon_cuda.py

.PHONY: install install-ref

install:
	mkdir -p $(PYBOX_DIR)
	cp $(PYBOX_FILES) $(PYBOX_DIR)/
	@echo "Installed pybox to $(PYBOX_DIR)"

# Copies bundled PyTorch reference inference code to /opt/corridorkey/reference/
# Works on any platform -- no external repo dependency.
install-ref:
	mkdir -p $(REF_DIR)
	cp -r ../reference/CorridorKeyModule $(REF_DIR)/
	cp -r ../reference/utils $(REF_DIR)/
	@echo "Installed reference inference code to $(REF_DIR)"
