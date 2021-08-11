#include "kmp.h"
#include <assert.h>
#include <limits>

namespace PollingAPI {

    //! \brief the parameters of the nanos6_register_polling_service function
    struct ServiceKey {
        nanos6_polling_service_t _function;
        void *_functionData;

        ServiceKey()
            :  _function(nullptr), _functionData(nullptr)
        { }

        ServiceKey(nanos6_polling_service_t function, void *functionData)
            :  _function(function), _functionData(functionData)
        { }

        void reset() {
            _function = nullptr;
            _functionData = nullptr;
        }
    };


    //! \brief the status of a service
    struct ServiceData {
        //! \brief Indicates whether the service is being processed at that moment
        bool _processing;

        //! \brief A pointer to an area that is set when the service has been
        //! marked for removal and that will be set to true once
        //! the service has been unregistered
        std::atomic<bool> *_discard;

        ServiceData()
        : _processing(false), _discard(nullptr)
        { }

        void reset() {
            _processing = false;
            _discard = nullptr;
        }
    };


    struct ServiceInfo {
        bool _valid;
        ServiceKey _key;
        ServiceData _data;

        ServiceInfo()
            : _valid(false), _key(), _data()
        { }

        void reset() {
            _valid = false;
            _key.reset();
            _data.reset();
        }
    };

    #define MAX_SERVICES 8

    //! \brief Services in the system
    std::atomic<unsigned int> _next_free_service_slot;
    ServiceInfo _services[MAX_SERVICES];

    class Spinlock {
    public:
        Spinlock() { m_lock.clear(); }
        Spinlock(const Spinlock&) = delete;
        ~Spinlock() = default;

        void lock() {
            while (m_lock.test_and_set(std::memory_order_acquire));
        }
        bool try_lock() {
            return !m_lock.test_and_set(std::memory_order_acquire);
        }
        void unlock() {
            m_lock.clear(std::memory_order_release);
        }
    private:
        std::atomic_flag m_lock;
    };

    //! \brief This is held during traversal and modification
    //! operations but not while processing a service
    Spinlock _lock;

    inline bool operator<(ServiceKey const &a, ServiceKey const &b)
    {
        if (a._function < b._function) {
            return true;
        } else if (a._function > b._function) {
            return false;
        }

        if (a._functionData < b._functionData) {
            return true;
        } else if (a._functionData > b._functionData) {
            return false;
        }

        return false; // Equal
    }
}

void *nanos6_get_current_event_counter() {
    kmp_int32 gtid = __kmp_entry_gtid();
    kmp_info_t *thread = __kmp_threads[gtid];
    kmp_taskdata_t *taskdata = thread->th.th_current_task;
    taskdata->td_flags.detachable = TASK_DETACHABLE;

    kmp_event_t *event_counter = &taskdata->td_allow_completion_event;

    KMP_DEBUG_ASSERT(event_counter->ed.task != nullptr);
    KMP_DEBUG_ASSERT(event_counter->pending_events_count != -1);

    KA_TRACE(5, ("event %p requested\n", event_counter));
    return event_counter;
}

void nanos6_increase_current_task_event_counter(void *event_counter,
                                                unsigned int increment) {
    // Atomic ops require signed, check if value is representable.
    KMP_DEBUG_ASSERT(sizeof(unsigned int) == sizeof(kmp_uint32));
    KMP_DEBUG_ASSERT(increment < std::numeric_limits<kmp_int32>::max());

    kmp_event_t *completion_event = static_cast<kmp_event_t *>(event_counter);

    KMP_DEBUG_ASSERT(completion_event->ed.task != nullptr);
    KMP_DEBUG_ASSERT(completion_event->pending_events_count != -1);

    kmp_uint32 pending_events_count
      = KMP_TEST_THEN_ADD32(&(completion_event->pending_events_count), increment);
    KA_TRACE(5, ("event %p pending_events %d -> %d\n",
              event_counter, pending_events_count, pending_events_count + increment));
}

void nanos6_decrease_task_event_counter(void *event_counter,
                                        unsigned int decrement) {
    // Atomic ops require signed, check if value is representable.
    KMP_DEBUG_ASSERT(sizeof(unsigned int) == sizeof(kmp_uint32));
    KMP_DEBUG_ASSERT(decrement < std::numeric_limits<kmp_int32>::max());

    kmp_event_t *completion_event = static_cast<kmp_event_t *>(event_counter);

    KMP_DEBUG_ASSERT(completion_event->ed.task != nullptr);
    KMP_DEBUG_ASSERT(completion_event->pending_events_count != -1);

    kmp_uint32 pending_events_count
      = KMP_TEST_THEN_ADD32(&(completion_event->pending_events_count), -decrement);
    KA_TRACE(5, ("event %p pending_events %d -> %d\n",
              event_counter, pending_events_count, pending_events_count - decrement));
    // 0 means task has finished and there is no mpi ops remaining
    if (pending_events_count == decrement) {
        kmp_task_t *ptask = completion_event->ed.task;
        kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(ptask);

        int gtid = __kmp_get_gtid();

        KMP_DEBUG_ASSERT(taskdata->td_flags.executing == 1);
        taskdata->td_flags.executing = 0; // suspend the finishing task
        taskdata->td_flags.proxy = TASK_PROXY; // proxify!

        // If the task detached complete the proxy task
        if (gtid >= 0) {
          kmp_team_t *team = taskdata->td_team;
          kmp_info_t *thread = __kmp_get_thread();
          if (thread->th.th_team == team) {
            __kmpc_proxy_task_completed(gtid, ptask);
            return;
          }
        }
        // fallback
        __kmpc_proxy_task_completed_ooo(ptask);
    }
}

void handleServices()
{
    // Nothing to do
    unsigned int num_slot = KMP_ATOMIC_LD_RLX(&PollingAPI::_next_free_service_slot);
    if (num_slot == 0)
        return;

    bool locked = PollingAPI::_lock.try_lock();
    if (!locked) {
        return;
    }

    for (unsigned int i = 0; i < num_slot; ++i)
    {
        // Ignore the current slot if the service that was registered there has been removed
        if (!PollingAPI::_services[i]._valid)
            continue;

        const PollingAPI::ServiceKey &serviceKey = PollingAPI::_services[i]._key;
        PollingAPI::ServiceData &serviceData = PollingAPI::_services[i]._data;

        // Somebody else processing it?
        if (serviceData._processing)
            continue;

        if (serviceData._discard != nullptr) {
            // Signal the unregistration
            serviceData._discard->store(true);

            PollingAPI::_services[i]._valid = false;
            PollingAPI::_services[i].reset();
            continue;
        }

        // Set the processing flag
        assert(!serviceData._processing);
        serviceData._processing = true;

        // Execute the callback without locking
        PollingAPI::_lock.unlock();
        bool unregister = serviceKey._function(serviceKey._functionData);
        PollingAPI::_lock.lock();

        // Unset the processing flag
        assert(serviceData._processing);
        serviceData._processing = false;

        // By construction, even in the presence of concurrent
        // calls to this method, the iterator remains valid

        // If the function returns true or the service had been marked
        // for unregistration, remove the service
        if (unregister || (serviceData._discard != nullptr)) {

            if (serviceData._discard != nullptr) {
                // Signal the unregistration
                serviceData._discard->store(true);
            }
            PollingAPI::_services[i]._valid = false;
            PollingAPI::_services[i].reset();
        }
    }

    PollingAPI::_lock.unlock();
}



void nanos6_register_polling_service(char const *service_name,
                                     nanos6_polling_service_t service_function,
                                     void *service_data)
{
    PollingAPI::_lock.lock();

    assert(PollingAPI::_next_free_service_slot < MAX_SERVICES && "Trying to register more services than available slots");

    PollingAPI::_services[PollingAPI::_next_free_service_slot]._key =
        PollingAPI::ServiceKey(service_function, service_data);

    PollingAPI::_services[PollingAPI::_next_free_service_slot]._valid = true;

    PollingAPI::_next_free_service_slot++;

    PollingAPI::_lock.unlock();
}

void nanos6_unregister_polling_service(char const *service_name,
                                       nanos6_polling_service_t service_function,
                                       void *service_data)
{
    std::atomic<bool> unregistered(false);

    PollingAPI::ServiceKey key(service_function, service_data);

    {
        PollingAPI::_lock.lock();

        bool found = false;
        for (unsigned int i = 0; i < PollingAPI::_next_free_service_slot && !found; ++i)
        {
            if (!PollingAPI::_services[i]._valid)
                continue;

           if (!(key < PollingAPI::_services[i]._key) && !(PollingAPI::_services[i]._key < key))
           {
                found = true;
                PollingAPI::ServiceData &serviceData = PollingAPI::_services[i]._data;

                assert((serviceData._discard == nullptr)
                        && "Attempt to unregister an already unregistered polling service");

                // Set up unregistering protocol
                serviceData._discard = &unregistered;
           }
        }
        assert(found && "Attempt to unregister a non-existing polling service");

        PollingAPI::_lock.unlock();
    }

    // Wait until fully unregistered
    while (unregistered.load() == false) {
            // Try to speed up the unregistration
            handleServices();
    }
}

// This is supposed to be called in a MPI_Init
void nanos6_notify_task_event_counter_api() {
    int gtid = __kmp_entry_gtid();
    __kmp_enable_tasking_in_serial_mode(
      NULL, gtid,
      /*proxy=*/false, /*detachable=*/true, /*hidden_helper=*/false);
    KA_TRACE(5, ("event tasking  enabled\n"));
}
