package config

import (
	"fmt"
	"net/url"
	"slices"
)

type PeerDictionaryConfig map[string]PeerConfig
type PeerConfig struct {
	Proxy    string   `yaml:"proxy"`
	ProxyURL *url.URL `yaml:"-"`
	ApiKey   string   `yaml:"apiKey"`
	Models   []string `yaml:"models"`
	Filters  Filters  `yaml:"filters"`
	// Modalities declares the input/output modalities of this peer's models,
	// for peers whose /v1/models carries no `architecture` block (plain vLLM,
	// speech servers). Keys must be listed in Models. What the peer advertises
	// itself always wins; a declaration only fills the gap.
	Modalities map[string]PeerModality `yaml:"modalities"`
}

// PeerModality is a declared modality signature, e.g. {input: [audio], output: [text]}.
type PeerModality struct {
	Input  []string `yaml:"input"`
	Output []string `yaml:"output"`
}

func (c *PeerConfig) UnmarshalYAML(unmarshal func(interface{}) error) error {
	type rawPeerConfig PeerConfig
	defaults := rawPeerConfig{
		Proxy:   "",
		ApiKey:  "",
		Models:  []string{},
		Filters: Filters{},
	}

	if err := unmarshal(&defaults); err != nil {
		return err
	}

	// Validate proxy is not empty
	if defaults.Proxy == "" {
		return fmt.Errorf("proxy is required")
	}

	// Validate proxy is a valid URL and store the parsed value
	parsedURL, err := url.Parse(defaults.Proxy)
	if err != nil {
		return fmt.Errorf("invalid peer proxy URL (%s): %w", defaults.Proxy, err)
	}
	defaults.ProxyURL = parsedURL

	// Validate models is not empty
	if len(defaults.Models) == 0 {
		return fmt.Errorf("peer models can not be empty")
	}

	for model, mod := range defaults.Modalities {
		if !slices.Contains(defaults.Models, model) {
			return fmt.Errorf("modalities: %q is not in this peer's models", model)
		}
		if len(mod.Input) == 0 && len(mod.Output) == 0 {
			return fmt.Errorf("modalities: %q declares neither input nor output", model)
		}
	}

	*c = PeerConfig(defaults)
	return nil
}
