export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "13.0.5"
  }
  public: {
    Tables: {
      ab_tests: {
        Row: {
          confidence_level: number | null
          conversion_goal: string
          created_at: string
          description: string | null
          ended_at: string | null
          id: string
          landing_page_id: string
          name: string
          started_at: string | null
          statistical_power: number | null
          status: string
          test_type: string
          traffic_allocation: Json | null
          updated_at: string
          user_id: string | null
          variants: Json
          winner_variant_id: string | null
        }
        Insert: {
          confidence_level?: number | null
          conversion_goal: string
          created_at?: string
          description?: string | null
          ended_at?: string | null
          id?: string
          landing_page_id: string
          name: string
          started_at?: string | null
          statistical_power?: number | null
          status?: string
          test_type?: string
          traffic_allocation?: Json | null
          updated_at?: string
          user_id?: string | null
          variants?: Json
          winner_variant_id?: string | null
        }
        Update: {
          confidence_level?: number | null
          conversion_goal?: string
          created_at?: string
          description?: string | null
          ended_at?: string | null
          id?: string
          landing_page_id?: string
          name?: string
          started_at?: string | null
          statistical_power?: number | null
          status?: string
          test_type?: string
          traffic_allocation?: Json | null
          updated_at?: string
          user_id?: string | null
          variants?: Json
          winner_variant_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "ab_tests_landing_page_id_fkey"
            columns: ["landing_page_id"]
            isOneToOne: false
            referencedRelation: "landing_pages"
            referencedColumns: ["id"]
          },
        ]
      }
      ai_agents: {
        Row: {
          agent_type: string
          campaign_id: string | null
          config: Json | null
          created_at: string
          expertise: string | null
          id: string
          is_active: boolean | null
          name: string
          personality: string | null
          prompt_template: string | null
          updated_at: string
        }
        Insert: {
          agent_type: string
          campaign_id?: string | null
          config?: Json | null
          created_at?: string
          expertise?: string | null
          id?: string
          is_active?: boolean | null
          name: string
          personality?: string | null
          prompt_template?: string | null
          updated_at?: string
        }
        Update: {
          agent_type?: string
          campaign_id?: string | null
          config?: Json | null
          created_at?: string
          expertise?: string | null
          id?: string
          is_active?: boolean | null
          name?: string
          personality?: string | null
          prompt_template?: string | null
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "ai_agents_campaign_id_fkey"
            columns: ["campaign_id"]
            isOneToOne: false
            referencedRelation: "ai_campaigns"
            referencedColumns: ["id"]
          },
        ]
      }
      ai_campaigns: {
        Row: {
          brand_guidelines: Json | null
          business_goals: string | null
          created_at: string
          description: string | null
          id: string
          industry: string | null
          name: string
          status: string
          target_audience: string | null
          tone_of_voice: string | null
          updated_at: string
        }
        Insert: {
          brand_guidelines?: Json | null
          business_goals?: string | null
          created_at?: string
          description?: string | null
          id?: string
          industry?: string | null
          name: string
          status?: string
          target_audience?: string | null
          tone_of_voice?: string | null
          updated_at?: string
        }
        Update: {
          brand_guidelines?: Json | null
          business_goals?: string | null
          created_at?: string
          description?: string | null
          id?: string
          industry?: string | null
          name?: string
          status?: string
          target_audience?: string | null
          tone_of_voice?: string | null
          updated_at?: string
        }
        Relationships: []
      }
      ai_concepts: {
        Row: {
          agent_id: string | null
          campaign_id: string | null
          components_needed: string[] | null
          concept_description: string | null
          concept_name: string
          created_at: string
          creativity_score: number | null
          estimated_conversion_rate: number | null
          feasibility_score: number | null
          id: string
          key_benefits: string[] | null
          metadata: Json | null
          psychological_triggers: string[] | null
          solution_approach: string | null
          status: string
          target_problem: string | null
          updated_at: string
        }
        Insert: {
          agent_id?: string | null
          campaign_id?: string | null
          components_needed?: string[] | null
          concept_description?: string | null
          concept_name: string
          created_at?: string
          creativity_score?: number | null
          estimated_conversion_rate?: number | null
          feasibility_score?: number | null
          id?: string
          key_benefits?: string[] | null
          metadata?: Json | null
          psychological_triggers?: string[] | null
          solution_approach?: string | null
          status?: string
          target_problem?: string | null
          updated_at?: string
        }
        Update: {
          agent_id?: string | null
          campaign_id?: string | null
          components_needed?: string[] | null
          concept_description?: string | null
          concept_name?: string
          created_at?: string
          creativity_score?: number | null
          estimated_conversion_rate?: number | null
          feasibility_score?: number | null
          id?: string
          key_benefits?: string[] | null
          metadata?: Json | null
          psychological_triggers?: string[] | null
          solution_approach?: string | null
          status?: string
          target_problem?: string | null
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "ai_concepts_agent_id_fkey"
            columns: ["agent_id"]
            isOneToOne: false
            referencedRelation: "ai_agents"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "ai_concepts_campaign_id_fkey"
            columns: ["campaign_id"]
            isOneToOne: false
            referencedRelation: "ai_campaigns"
            referencedColumns: ["id"]
          },
        ]
      }
      ai_generated_components: {
        Row: {
          agent_id: string | null
          api_endpoints: string[] | null
          code_structure: Json | null
          component_name: string
          component_type: string
          concept_id: string | null
          created_at: string
          dependencies: string[] | null
          description: string | null
          functionality: string | null
          generated_code: string | null
          id: string
          implementation_complexity: number | null
          metadata: Json | null
          status: string
          styling_requirements: string | null
          updated_at: string
        }
        Insert: {
          agent_id?: string | null
          api_endpoints?: string[] | null
          code_structure?: Json | null
          component_name: string
          component_type: string
          concept_id?: string | null
          created_at?: string
          dependencies?: string[] | null
          description?: string | null
          functionality?: string | null
          generated_code?: string | null
          id?: string
          implementation_complexity?: number | null
          metadata?: Json | null
          status?: string
          styling_requirements?: string | null
          updated_at?: string
        }
        Update: {
          agent_id?: string | null
          api_endpoints?: string[] | null
          code_structure?: Json | null
          component_name?: string
          component_type?: string
          concept_id?: string | null
          created_at?: string
          dependencies?: string[] | null
          description?: string | null
          functionality?: string | null
          generated_code?: string | null
          id?: string
          implementation_complexity?: number | null
          metadata?: Json | null
          status?: string
          styling_requirements?: string | null
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "ai_generated_components_agent_id_fkey"
            columns: ["agent_id"]
            isOneToOne: false
            referencedRelation: "ai_agents"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "ai_generated_components_concept_id_fkey"
            columns: ["concept_id"]
            isOneToOne: false
            referencedRelation: "ai_concepts"
            referencedColumns: ["id"]
          },
        ]
      }
      analytics_events: {
        Row: {
          ab_test_id: string | null
          created_at: string
          event_data: Json | null
          event_type: string
          id: string
          ip_address: unknown | null
          landing_page_id: string | null
          referrer: string | null
          session_id: string
          user_agent: string | null
          user_id: string | null
          variant_id: string | null
        }
        Insert: {
          ab_test_id?: string | null
          created_at?: string
          event_data?: Json | null
          event_type: string
          id?: string
          ip_address?: unknown | null
          landing_page_id?: string | null
          referrer?: string | null
          session_id: string
          user_agent?: string | null
          user_id?: string | null
          variant_id?: string | null
        }
        Update: {
          ab_test_id?: string | null
          created_at?: string
          event_data?: Json | null
          event_type?: string
          id?: string
          ip_address?: unknown | null
          landing_page_id?: string | null
          referrer?: string | null
          session_id?: string
          user_agent?: string | null
          user_id?: string | null
          variant_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "analytics_events_ab_test_id_fkey"
            columns: ["ab_test_id"]
            isOneToOne: false
            referencedRelation: "ab_tests"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "analytics_events_landing_page_id_fkey"
            columns: ["landing_page_id"]
            isOneToOne: false
            referencedRelation: "landing_pages"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "analytics_events_variant_id_fkey"
            columns: ["variant_id"]
            isOneToOne: false
            referencedRelation: "test_variants"
            referencedColumns: ["id"]
          },
        ]
      }
      campaign_analytics: {
        Row: {
          campaign_id: string | null
          concept_id: string | null
          id: string
          metadata: Json | null
          metric_name: string
          metric_type: string
          metric_value: number | null
          recorded_at: string
        }
        Insert: {
          campaign_id?: string | null
          concept_id?: string | null
          id?: string
          metadata?: Json | null
          metric_name: string
          metric_type: string
          metric_value?: number | null
          recorded_at?: string
        }
        Update: {
          campaign_id?: string | null
          concept_id?: string | null
          id?: string
          metadata?: Json | null
          metric_name?: string
          metric_type?: string
          metric_value?: number | null
          recorded_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "campaign_analytics_campaign_id_fkey"
            columns: ["campaign_id"]
            isOneToOne: false
            referencedRelation: "ai_campaigns"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "campaign_analytics_concept_id_fkey"
            columns: ["concept_id"]
            isOneToOne: false
            referencedRelation: "ai_concepts"
            referencedColumns: ["id"]
          },
        ]
      }
      integrations: {
        Row: {
          config: Json
          created_at: string
          id: string
          is_active: boolean | null
          provider: string
          updated_at: string
          user_id: string | null
        }
        Insert: {
          config?: Json
          created_at?: string
          id?: string
          is_active?: boolean | null
          provider: string
          updated_at?: string
          user_id?: string | null
        }
        Update: {
          config?: Json
          created_at?: string
          id?: string
          is_active?: boolean | null
          provider?: string
          updated_at?: string
          user_id?: string | null
        }
        Relationships: []
      }
      landing_pages: {
        Row: {
          components: Json
          config: Json
          conversion_rate: number | null
          created_at: string
          id: string
          seo_config: Json | null
          slug: string
          status: string
          template_type: string
          title: string
          traffic_count: number | null
          updated_at: string
          user_id: string | null
        }
        Insert: {
          components?: Json
          config?: Json
          conversion_rate?: number | null
          created_at?: string
          id?: string
          seo_config?: Json | null
          slug: string
          status?: string
          template_type?: string
          title: string
          traffic_count?: number | null
          updated_at?: string
          user_id?: string | null
        }
        Update: {
          components?: Json
          config?: Json
          conversion_rate?: number | null
          created_at?: string
          id?: string
          seo_config?: Json | null
          slug?: string
          status?: string
          template_type?: string
          title?: string
          traffic_count?: number | null
          updated_at?: string
          user_id?: string | null
        }
        Relationships: []
      }
      profiles: {
        Row: {
          avatar_url: string | null
          created_at: string
          email: string | null
          full_name: string | null
          id: string
          role: string | null
          updated_at: string
          user_id: string
        }
        Insert: {
          avatar_url?: string | null
          created_at?: string
          email?: string | null
          full_name?: string | null
          id?: string
          role?: string | null
          updated_at?: string
          user_id: string
        }
        Update: {
          avatar_url?: string | null
          created_at?: string
          email?: string | null
          full_name?: string | null
          id?: string
          role?: string | null
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      test_variants: {
        Row: {
          ab_test_id: string
          config: Json
          conversion_rate: number | null
          conversions: number | null
          created_at: string
          description: string | null
          id: string
          is_control: boolean | null
          name: string
          traffic_percentage: number | null
          updated_at: string
          visitors: number | null
        }
        Insert: {
          ab_test_id: string
          config?: Json
          conversion_rate?: number | null
          conversions?: number | null
          created_at?: string
          description?: string | null
          id?: string
          is_control?: boolean | null
          name: string
          traffic_percentage?: number | null
          updated_at?: string
          visitors?: number | null
        }
        Update: {
          ab_test_id?: string
          config?: Json
          conversion_rate?: number | null
          conversions?: number | null
          created_at?: string
          description?: string | null
          id?: string
          is_control?: boolean | null
          name?: string
          traffic_percentage?: number | null
          updated_at?: string
          visitors?: number | null
        }
        Relationships: [
          {
            foreignKeyName: "test_variants_ab_test_id_fkey"
            columns: ["ab_test_id"]
            isOneToOne: false
            referencedRelation: "ab_tests"
            referencedColumns: ["id"]
          },
        ]
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {},
  },
} as const
